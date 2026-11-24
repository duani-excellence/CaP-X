"""Generic public RGB-D instance probes, shared by all native task adapters.

Visible point-cloud bounds are not full shape or insertion alignment evidence.
No privileged fallback, evaluator result, task identifier or simulator state is
used by localization or contract scheduling.
"""
from pathlib import Path
import copy
import time
import numpy as np
from capx.experiments.native_tasks import save
from capx.execution.perception_evidence import ExactEvidenceCache,current_context,evidence_scope


class PublicScene:
    """Multi-view instance binding using only supplied public sensor callbacks.

    Region roles are initialized by observed loose-object membership, then
    associated geometrically. This class has no simulator or evaluator handle.
    A shape here is a measurement, not a qualified release certificate.
    """
    def __init__(self,capture,segment,bindings,*,region_query='wooden tray',emit=None,optimize=False):
        self.capture=capture;self.segment=segment;self.bindings=bindings
        self.region_query=region_query;self.emit=emit or (lambda *a,**k:None)
        self.roles=None;self.serial=0;self.cache={};self.source=None
        self.optimize=optimize;self.semantic_memory=ExactEvidenceCache();self.query_serial=0
        self.geometry_cache={};self.object_cache={};self.observation_key=None

    @evidence_scope
    def frame(self):
        started=time.monotonic();self.obs,self.source=self.capture();captured=time.monotonic()
        self.serial+=1;self.cache={}
        self.geometry_cache={};self.object_cache={};self.observation_key=ExactEvidenceCache.digest(self.obs)
        self.emit('perception_frame',source=self.source,observation_key=self.observation_key,context=current_context(),
            capture_and_archive_seconds=captured-started,identity_seconds=time.monotonic()-captured)
        return self.obs

    def _segment(self,image,label,camera,crop=None):
        self.query_serial+=1;start=time.monotonic()
        calibration=ExactEvidenceCache.digest({k:v for k,v in self.obs[camera].items() if k!='images'})
        image_hash=ExactEvidenceCache.digest(image);key=(camera,calibration,label,image_hash)
        compute=lambda:self.segment(image,text_prompt=label)
        raw=None;hit=False;origin=self.source;error=None
        try:
            if self.optimize:raw,hit,origin=self.semantic_memory.get_or_compute(key,compute,self.source)
            else:raw,hit,origin=compute(),False,self.source
            return raw
        except Exception as exc:error=type(exc).__name__;raise
        finally:
            self.emit('perception_request',query_id=self.query_serial,tool='sam3',label=label,camera=camera,
                crop=crop,image_hash=image_hash,calibration_hash=calibration,source=self.source,
                reused=hit,origin=origin,seconds=time.monotonic()-start,context=current_context(),error=error,
                result_count=len(raw) if raw is not None else None,result_hash=ExactEvidenceCache.digest(raw),
                reuse_basis='identical_current_RGB_and_calibration' if hit else None)

    @evidence_scope
    def candidates(self,label,region=None):
        from capx.execution.region_geometry import instance_masks,deproject,horizontal_pixels,axis_pixels
        if isinstance(label,list):
            for text in label:
                rows=self.candidates(text,region)
                if region is not None:rows=[r for r in rows if self.in_region(r,region)]
                if not rows:continue
                centers=np.array([np.median(r['points'],axis=0) for r in rows])
                if np.max(np.linalg.norm(centers[:,None]-centers[None,:],axis=-1))<=.08:
                    self.emit('perception_binding',labels=label,selected=text,source=self.source,context=current_context())
                    return rows
            self.emit('perception_binding',labels=label,selected=None,source=self.source,context=current_context())
            return []
        key=(label,None if region is None else (ExactEvidenceCache.digest(region) if self.optimize else tuple(region['center'])))
        if key in self.cache:return self.cache[key]
        result=[]
        for name,camera in self.obs.items():
            if not isinstance(camera,dict) or 'images' not in camera:continue
            rgb=camera['images']['rgb'];raw=self._segment(rgb,label,name)
            if region is not None:
                import itertools
                # Public receptacle-guided zoom changes observability, not the
                # confidence threshold. No private box or object pose is used.
                corners=np.array(list(itertools.product(*zip(region['lo'],region['hi']),
                                     [region['floor_z'],region['floor_z']+.25])))
                t=camera['pose_mat'];p=(corners-t[:3,3])@t[:3,:3];q=p@camera['intrinsics'].T
                uv=q[:,:2]/q[:,2,None]
                if np.all(p[:,2]>0):
                    a=np.maximum(0,np.floor(uv.min(0))-8).astype(int)
                    b=np.minimum([rgb.shape[1],rgb.shape[0]],np.ceil(uv.max(0))+8).astype(int)
                    if np.all(b-a>32):
                        crop=rgb[a[1]:b[1],a[0]:b[0]]
                        for row in self._segment(crop,label,name,[a.tolist(),b.tolist()]):
                            mask=np.zeros(rgb.shape[:2],bool);mask[a[1]:b[1],a[0]:b[0]]=row['mask']
                            raw.append(dict(row,mask=mask))
                        self.emit('public_region_crop',label=label,camera=name,box=[a.tolist(),b.tolist()],source=self.source)
            if self.optimize:
                if name not in self.geometry_cache:
                    xyz=deproject(camera)
                    self.geometry_cache[name]=(xyz,horizontal_pixels(xyz),[axis_pixels(xyz,a) for a in (0,1)])
                xyz,horizontal,axes=self.geometry_cache[name]
            else:xyz=deproject(camera)
            depth=camera['images']['depth'].squeeze()
            for group in instance_masks(raw):
                mask=group['mask']&(depth>0)&np.isfinite(xyz).all(-1);points=xyz[mask]
                if len(points)<30:continue
                result.append(dict(camera=name,mask=mask,points=points,
                    floor_points=xyz[mask&(horizontal if self.optimize else horizontal_pixels(xyz))],
                    walls=[(a,xyz[mask&(axes[a] if self.optimize else axis_pixels(xyz,a))]) for a in (0,1)],scores=group['scores'],source=self.source))
            self.emit('instance_query',label=label,camera=name,raw_scores=[float(r['score']) for r in raw],source=self.source)
        self.cache[key]=result
        return result

    def regions(self):
        from capx.execution.region_geometry import horizontal_region,same_region,receptacle_geometry
        candidates=[]
        for row in self.candidates(self.region_query):
            fit=horizontal_region(row['floor_points'])
            if fit['status']!='PASS':continue
            duplicate=next((r for r in candidates if same_region(r,fit)),None)
            if duplicate is None:candidates.append(dict(fit,rows=[row]))
            else:duplicate['rows'].append(row)
        if len(candidates)!=2:raise ValueError('Two uniquely observed receptacles required')
        result=[]
        for candidate in candidates:
            rows=candidate['rows']
            fit=receptacle_geometry(np.concatenate([r['floor_points'] for r in rows]),
                                    [w for r in rows for w in r['walls']])
            if fit['status']!='PASS':raise ValueError('Unbounded receptacle: '+fit['reason'])
            result.append(dict(fit,source=self.source,cameras=[r['camera'] for r in rows]))
        return result

    @staticmethod
    def in_region(row,region):
        p=row['points'];lo=np.asarray(region['lo']);hi=np.asarray(region['hi'])
        keep=np.all((p[:,:2]>=lo-.01)&(p[:,:2]<=hi+.01),axis=1)
        if 'ceiling_z' in region:keep &= (p[:,2]>=region['floor_z'])&(p[:,2]<=region['ceiling_z'])
        return float(np.mean(keep))>.85

    @evidence_scope
    def bind_regions(self):
        regions=self.regions()
        if self.roles is None:
            counts=[sum(any(self.in_region(c,r) for c in self.candidates(label)) for label in self.bindings.values()) for r in regions]
            if max(counts)<2 or counts[0]==counts[1]:raise ValueError('Receptacle role ambiguity')
            index=int(np.argmax(counts));self.roles={'source':regions[index],'destination':regions[1-index]}
        else:
            matched={}
            for role,old in self.roles.items():
                distances=[np.linalg.norm(np.asarray(r['center'])-old['center']) for r in regions]
                i=int(np.argmin(distances))
                if distances[i]>.03:raise ValueError('Receptacle moved or current evidence contradicts role anchor')
                matched[role]=regions[i]
            if matched['source'] is matched['destination']:raise ValueError('Receptacle association collision')
            self.roles=matched
        self.emit('region_roles',roles=self.roles,source=self.source)
        return self.roles

    @evidence_scope
    def observe_region(self,role):
        """Revalidate one known role with current measurements, not all roles.

        The old bound proposes association only. It cannot fill missing walls;
        an unmeasured/moved fixture still fails the same geometric fit.
        """
        from capx.execution.region_geometry import receptacle_geometry
        if self.roles is None or role not in self.roles:raise ValueError('Unbound region role')
        old=self.roles[role];floors=[];walls=[]
        def selected(points):
            p=np.asarray(points);keep=np.all((p[:,:2]>=np.asarray(old['lo'])-.01)&(p[:,:2]<=np.asarray(old['hi'])+.01),axis=1)
            return p[keep]
        for row in self.candidates(self.region_query):
            floors.append(selected(row['floor_points']))
            walls.extend((axis,selected(points)) for axis,points in row['walls'])
        if not floors:raise ValueError('No current region evidence')
        fit=receptacle_geometry(np.concatenate(floors),walls)
        if fit['status']!='PASS':raise ValueError('Region evidence incomplete: '+fit['reason'])
        if np.linalg.norm(np.asarray(fit['center'])-old['center'])>.01:raise ValueError('Region anchor contradicted')
        self.roles[role]=dict(fit,source=self.source)
        self.emit('single_region_revalidation',role=role,geometry=self.roles[role],source=self.source)
        return self.roles[role]

    @evidence_scope
    def object(self,name,region=None):
        key=(name,ExactEvidenceCache.digest(region))
        if self.optimize and key in self.object_cache:
            self.emit('perception_geometry_reuse',object=name,source=self.source,kind_of_reuse='same_current_frame_point_cloud',context=current_context())
            return copy.deepcopy(self.object_cache[key])
        result=self._object_current(name,region)
        if self.optimize:self.object_cache[key]=copy.deepcopy(result)
        return result

    def _object_current(self,name,region=None):
        from capx.execution.region_geometry import upright_shape,foreground_component
        rows=self.candidates(self.bindings[name],region)
        if region is not None:rows=[r for r in rows if self.in_region(r,region)]
        if not rows:raise ValueError('No current unambiguous instance for '+name)
        # Candidate centers must agree across views. A label is not an instance
        # selector; distinct clusters are rejected instead of choosing a score.
        centers=np.array([np.median(r['points'],axis=0) for r in rows])
        if np.max(np.linalg.norm(centers[:,None]-centers[None,:],axis=-1))>.08:
            raise ValueError('Distinct candidates require additional role evidence')
        points,screen=foreground_component(np.concatenate([r['points'] for r in rows]));floor=region['floor_z'] if region is not None else float(np.quantile(points[:,2],.005))
        fit=upright_shape(points,floor)
        self.emit('object_geometry',object=name,geometry=fit,source=self.source,cameras=[r['camera'] for r in rows],component_screen=screen)
        return dict(fit,points=points,views=rows,source=self.source)


def select_instances(rows,minimum_score=.5,iou_threshold=.7):
    """Confidence-filter and deduplicate masks; never pick among distinct instances.

    A score is a tool screening heuristic, NOT a calibrated pose uncertainty.
    These fixed development settings do not authorize any physical action.
    """
    selected=[]
    for row in sorted(rows,key=lambda r:r['score'],reverse=True):
        if not np.isfinite(row['score']) or row['score']<minimum_score:continue
        mask=np.asarray(row['mask'],bool)
        if not mask.any():continue
        duplicate=any(np.sum(mask&np.asarray(r['mask'],bool))/max(1,np.sum(mask|np.asarray(r['mask'],bool)))>iou_threshold for r in selected)
        if not duplicate:selected.append(row)
    return selected


def localize(boundary,label,camera):
    from capx.execution.public_camera import robosuite_pixel_centers
    from capx.experiments.vision_tasks import public_depth_sampling
    raw=boundary.capture()
    obs,calibration=robosuite_pixel_centers(raw,depth_sampling=public_depth_sampling(boundary),require_depth_sampling=True)
    cam=obs[camera];rgb=cam['images']['rgb'];depth=np.asarray(cam['images']['depth']).squeeze()
    started=time.monotonic();detections=boundary.api.sam3_seg_fn(rgb,text_prompt=label)
    k=cam['intrinsics'];t=cam['pose_mat'];yy,xx=np.indices(depth.shape)
    xyz=np.stack([(xx-k[0,2])*depth/k[0,0],(yy-k[1,2])*depth/k[1,1],depth],axis=-1)
    xyz=xyz@t[:3,:3].T+t[:3,3]
    candidates=[];capture=boundary.captures-1
    for i,item in enumerate(select_instances(detections)):
        mask=np.asarray(item['mask'],bool);points=xyz[mask&np.isfinite(xyz).all(axis=-1)&(depth>0)]
        if len(points)<30:continue
        lo,hi=np.quantile(points,[.01,.99],axis=0)
        path=boundary.root/'observations'/f'{capture:05d}-{camera}-{i}-mask.npz'
        # One capture per query, immutable evidence with calibrated source IDs.
        np.savez_compressed(path,mask=mask,points=points)
        candidates.append(dict(score=float(item['score']),pixels=len(points),lo=lo.tolist(),hi=hi.tolist(),
            center=((lo+hi)/2).tolist(),evidence=str(path),camera=camera,capture=capture,
            observation=str(boundary.root/'observations'/f'{capture:05d}-{camera}.npz'),
            kind='visible_surface_ONLY',calibration=calibration[camera]))
    result=dict(label=label,camera=camera,capture=capture,candidates=candidates,
        status='PASS' if len(candidates)==1 else 'UNCERTAIN',
        reason='one_visible_instance' if len(candidates)==1 else 'missing_or_ambiguous_instance',
        raw_detection_scores=[float(r['score']) for r in detections],
        uncertainty='no_full_shape_or_pose_error_bound',seconds=time.monotonic()-started)
    boundary.log('L0_perception',**result)
    return result


def probe(boundary):
    """Gate 3 diagnostics only: tool response is NOT task/skill acceptance."""
    queries=boundary.spec.get('perception_queries',{})
    if not queries:raise ValueError('No public object/fixture descriptions declared')
    result={};grasps={}
    for name,label in queries.items():
        result[name]=[]
        for camera in boundary.low.render_camera_names:
            try:result[name].append(localize(boundary,label,camera))
            except Exception as e:result[name].append(dict(status='FAIL',error=type(e).__name__+': '+str(e),camera=camera))
    for name in boundary.spec.get('objects',[]):
        if name not in queries:continue
        try:
            p,q=boundary.call('sample_grasp_pose',[queries[name]],{})
            grasps[name]=dict(status='candidate_only',position=p,quaternion=q)
        except Exception as e:grasps[name]=dict(status='FAIL',error=type(e).__name__+': '+str(e))
    save(boundary.root/'perception.json',dict(localization=result,grasps=grasps))
    return dict(passed=False,tool_probe_completed=True,
        localized=sum(any(v.get('status')=='PASS' for v in values) for values in result.values()),
        total_instances=len(result),grasp_candidates=sum(v['status']=='candidate_only' for v in grasps.values()),
        reason='Tool/instance evidence only; complete pose, skill safety and physical success still require separate gates')


def run_graph(boundary,spec):
    """Use existing GraphExecutive on declarative capabilities; fail closed.

    Registration of an unsupported skill never grants motion. This entry may
    collect only missing evidence until the declared primitive is qualified.
    """
    from capx.execution.skill_contracts import ContractGraph
    from capx.execution.graph_executive import GraphExecutive
    definition=spec.get('contract_task')
    if definition is None:raise ValueError('No contract task declaration')
    c=ContractGraph(definition);executive=GraphExecutive();decisions=[];excluded=[];attempts={}
    per_intent=spec.get('max_evidence_rounds',2)
    for _ in range(per_intent*len(c.intents)):
        decision=executive.next_contract(c,excluded=excluded);decisions.append(decision)
        if decision['work']=='stop' or decision['work']=='verify_task':break
        intent=decision['intent'];attempts[intent]=attempts.get(intent,0)+1
        if decision['work']=='skill':
            # The graph selects an eligible intent, but the capability registry
            # remains authoritative. Never fall through to raw robot motion.
            excluded.append(intent);decisions.append(dict(intent=intent,work='blocked',reason='unqualified_physical_skill'));continue
        useful=False
        for key in decision['missing']:
            query=spec.get('evidence_queries',{}).get(key)
            if query is None:continue
            evidence=localize(boundary,query['label'],query['camera'])
            old=c.truth(key);truth='TRUE' if evidence['status']=='PASS' and key.startswith('visible:') else 'UNKNOWN'
            c.observe(key,truth,evidence)
            useful |= old!=truth
        if not useful or attempts[intent]>=per_intent:excluded.append(intent)
    decision=executive.next_contract(c,excluded=excluded);decisions.append(decision)
    save(boundary.root/'contract-graph.json',dict(nodes=c.graph.nodes,edges=c.graph.edges,events=c.graph.events,
        decisions=decisions,complexity=c.complexity()))
    return dict(passed=False,graph_mechanism=True,decisions=decisions,complexity=c.complexity(),
        reason='No qualified non-cube physical Skill; unresolved contracts block actuation',new_llm_requests=0)
