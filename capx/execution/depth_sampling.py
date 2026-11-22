"""Measure a renderer's public depth sampling, without touching its scene FBO.

The probe writes only synthetic depths into private scratch renderbuffers. It
never reads model objects, physics, scene pixels, task success or API secrets.
"""
from __future__ import annotations

import copy
import hashlib
import json
import numpy as np

SCHEMA = 'capx.public_depth_sampling.v1'
SOURCE = 'https://github.com/google-deepmind/mujoco/blob/3.5.0/src/render/render_gl2.c'


def resolve_profile(sample_positions, trials, *, sample_count, descriptor=None):
    """Validate measured fixed-sample resolve; all probe pixels must agree.

    Each sample must have occupied every depth rank. This rejects min/max and
    averaged resolves instead of choosing whichever sample fits one scene.
    """
    base = dict(schema=SCHEMA, descriptor=copy.deepcopy(descriptor or {}),
                sample_count=int(sample_count), source_url=SOURCE,
                scope='synthetic scratch framebuffer; no task scene or physics',
                fitted_to_scene=False)
    def unknown(reason):
        return {**base, 'status':'UNCERTAIN', 'reason':reason}
    n = int(sample_count)
    if n == 0:
        result = {**base, 'status':'PASS', 'resolve_rule':'single_sample_pixel_center',
                  'selected_sample':None, 'sample_positions_bottom_origin_xy':[],
                  'depth_offset_from_pixel_center_xy':[0., 0.], 'probe_trials':[]}
    else:
        positions = np.asarray(sample_positions, float)
        if n < 1 or n > 16 or positions.shape != (n, 2) or not np.isfinite(positions).all() or np.any(positions < 0) or np.any(positions > 1):
            return unknown('invalid_or_unsupported_sample_positions')
        selected = None; ranks = [set() for _ in range(n)]; recorded=[]
        for trial in trials:
            assigned = np.asarray(trial.get('assigned_sample_depths', []), float)
            resolved = np.asarray(trial.get('resolved_depths', []), float)
            if assigned.shape != (n,) or resolved.size < 3 or not np.isfinite(assigned).all() or not np.isfinite(resolved).all() or len(np.unique(assigned)) != n:
                return unknown('invalid_synthetic_depth_trial')
            matches = np.flatnonzero(np.all(np.abs(assigned[:,None]-resolved.reshape(1,-1)) < 1e-5, axis=1))
            if len(matches) != 1 or (selected is not None and selected != int(matches[0])):
                return unknown('depth_resolve_is_not_one_fixed_sample')
            selected = int(matches[0])
            order=np.argsort(np.argsort(assigned))
            for i, rank in enumerate(order):ranks[i].add(int(rank))
            recorded.append({'assigned_sample_depths':assigned.tolist(),'resolved_depths':resolved.tolist()})
        if selected is None or any(len(r) != n for r in ranks):
            return unknown('insufficient_depth_rank_permutations')
        x,y = positions[selected]
        result={**base, 'status':'PASS', 'resolve_rule':'measured_fixed_depth_sample',
                'selected_sample':selected, 'sample_positions_bottom_origin_xy':positions.tolist(),
                'depth_offset_from_pixel_center_xy':[float(x-.5),float(.5-y)],
                'probe_trials':recorded,
                'rgb_depth_identity_limit':'RGB may average multiple samples; depth is one measured sample; a color edge is not an independent identity proof'}
    result['profile_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result


class DepthSamplingProbe:
    """Per-environment cache keyed by actual context and framebuffer resources."""
    def __init__(self, *, backend=None):
        self._backend=backend
        self._key=None;self._profile=None;self._owners=None

    def measure(self, render_context):
        try:
            if render_context is None or getattr(render_context,'con',None) is None or getattr(render_context,'gl_ctx',None) is None:
                raise ValueError('renderer_unavailable')
            backend=self._backend or _OpenGLProbe()
            render_context.gl_ctx.make_current()
            descriptor=backend.describe(render_context.con)
            owners=(render_context,render_context.con,render_context.gl_ctx)
            key=(tuple(id(x) for x in owners),json.dumps(descriptor,sort_keys=True))
            if key != self._key:
                n=descriptor['samples']
                if n:
                    positions,trials=backend.sample(descriptor)
                else:positions,trials=[],[]
                profile=resolve_profile(positions,trials,sample_count=n,descriptor=descriptor)
                self._key=key;self._owners=owners;self._profile=profile
            return copy.deepcopy(self._profile)
        except Exception as exc:
            # Do not return exception messages, model paths or opaque driver data.
            return dict(schema=SCHEMA,status='UNCERTAIN',reason='renderer_depth_calibration_unavailable',
                        error_type=type(exc).__name__,fitted_to_scene=False)


class _OpenGLProbe:
    def __init__(self):
        from OpenGL import GL
        self.gl=GL

    def describe(self, con):
        import mujoco
        # The source-level readPixels contract was audited for this installed
        # version; another implementation must be reviewed before authorization.
        if mujoco.__version__!='3.5.0':raise ValueError('unverified_mujoco_readpixels_contract')
        g=self.gl
        read=int(g.glGetIntegerv(g.GL_READ_FRAMEBUFFER_BINDING));draw=int(g.glGetIntegerv(g.GL_DRAW_FRAMEBUFFER_BINDING));rb=int(g.glGetIntegerv(g.GL_RENDERBUFFER_BINDING))
        try:
            fbo=int(con.offFBO)
            if not fbo:raise ValueError('offscreen_framebuffer_required')
            g.glBindFramebuffer(g.GL_FRAMEBUFFER,fbo)
            if g.glCheckFramebufferStatus(g.GL_FRAMEBUFFER)!=g.GL_FRAMEBUFFER_COMPLETE:raise ValueError('incomplete_live_framebuffer')
            attachments={}
            for label,attachment in [('depth',g.GL_DEPTH_ATTACHMENT),('color',g.GL_COLOR_ATTACHMENT0)]:
                kind=int(g.glGetFramebufferAttachmentParameteriv(g.GL_FRAMEBUFFER,attachment,g.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE))
                if kind!=g.GL_RENDERBUFFER:raise ValueError('unsupported_live_attachment')
                resource=int(g.glGetFramebufferAttachmentParameteriv(g.GL_FRAMEBUFFER,attachment,g.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME))
                g.glBindRenderbuffer(g.GL_RENDERBUFFER,resource)
                attachments[label]={key:int(g.glGetRenderbufferParameteriv(g.GL_RENDERBUFFER,enum)) for key,enum in [('format',g.GL_RENDERBUFFER_INTERNAL_FORMAT),('width',g.GL_RENDERBUFFER_WIDTH),('height',g.GL_RENDERBUFFER_HEIGHT),('samples',g.GL_RENDERBUFFER_SAMPLES)]}
                attachments[label]['resource']=resource
            depth,color=attachments['depth'],attachments['color']
            n=int(g.glGetIntegerv(g.GL_SAMPLES))
            if depth['samples']!=n or color['samples']!=n or n!=int(con.offSamples) or any(depth[k]!=color[k] for k in ('width','height')) or n<0 or n>16:raise ValueError('inconsistent_framebuffer_configuration')
            # mjr_render can change the clip-depth convention on this same
            # context (MuJoCo 3.5 uses reversed-Z with ZERO_TO_ONE). Record it
            # in the cache key and generate synthetic depths in that convention.
            # Explicit scalar buffers avoid PyOpenGL's incorrect auto-size
            # table for these newer scalar enums in the installed bindings.
            origin_value=np.zeros(1,dtype=np.int32);clip_value=np.zeros(1,dtype=np.int32)
            g.glGetIntegerv(g.GL_CLIP_ORIGIN,origin_value)
            g.glGetIntegerv(g.GL_CLIP_DEPTH_MODE,clip_value)
            origin=int(origin_value[0]);clip=int(clip_value[0])
            if origin!=g.GL_LOWER_LEFT:raise ValueError('unsupported_clip_origin')
            modes={int(g.GL_NEGATIVE_ONE_TO_ONE):'GL_NEGATIVE_ONE_TO_ONE',int(g.GL_ZERO_TO_ONE):'GL_ZERO_TO_ONE'}
            if clip not in modes:raise ValueError('unsupported_clip_depth_mode')
            positions=[np.asarray(g.glGetMultisamplefv(g.GL_SAMPLE_POSITION,i),float).tolist() for i in range(n)]
            def string(enum):
                value=g.glGetString(enum)
                if not value:raise ValueError('current_GL_context_required')
                return value.decode('ascii',errors='replace')
            return dict(framebuffer=fbo,width=depth['width'],height=depth['height'],samples=n,
                        context_width=int(con.offWidth),context_height=int(con.offHeight),attachments=attachments,
                        clip_origin='GL_LOWER_LEFT',clip_depth_mode=modes[clip],
                        sample_positions_bottom_origin_xy=positions,
                        mujoco_version=mujoco.__version__,reader_contract='mjr_readPixels_offscreen_color_depth_nearest_blit',
                        vendor=string(g.GL_VENDOR),renderer=string(g.GL_RENDERER),gl_version=string(g.GL_VERSION),
                        resolve_mask='GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT',resolve_filter='GL_NEAREST')
        finally:
            g.glBindRenderbuffer(g.GL_RENDERBUFFER,rb)
            g.glBindFramebuffer(g.GL_READ_FRAMEBUFFER,read);g.glBindFramebuffer(g.GL_DRAW_FRAMEBUFFER,draw)

    def sample(self, descriptor):
        """Same attachment formats/samples and blit as mjr_readPixels; scratch only."""
        g=self.gl;n=descriptor['samples'];size=8
        clip_mode=descriptor.get('clip_depth_mode')
        if clip_mode not in ('GL_NEGATIVE_ONE_TO_ONE','GL_ZERO_TO_ONE'):
            raise ValueError('unmeasured_clip_depth_mode')
        read=int(g.glGetIntegerv(g.GL_READ_FRAMEBUFFER_BINDING));draw=int(g.glGetIntegerv(g.GL_DRAW_FRAMEBUFFER_BINDING));rb=int(g.glGetIntegerv(g.GL_RENDERBUFFER_BINDING))
        program=int(g.glGetIntegerv(g.GL_CURRENT_PROGRAM));matrix_mode=int(g.glGetIntegerv(g.GL_MATRIX_MODE))
        mask_words=int(g.glGetIntegerv(g.GL_MAX_SAMPLE_MASK_WORDS))
        masks=[int(g.glGetIntegeri_v(g.GL_SAMPLE_MASK_VALUE,i)) for i in range(mask_words)]
        sample_mask_enabled=bool(g.glIsEnabled(g.GL_SAMPLE_MASK))
        fbos=[];buffers=[];pushed=False;matrices=[]
        try:
            g.glPushAttrib(g.GL_ALL_ATTRIB_BITS);pushed=True
            g.glUseProgram(0)
            for mode in (g.GL_PROJECTION,g.GL_MODELVIEW):
                g.glMatrixMode(mode);g.glPushMatrix();matrices.append(mode);g.glLoadIdentity()
            for index,samples in enumerate((n,0)):
                fbo=int(g.glGenFramebuffers(1));fbos.append(fbo);g.glBindFramebuffer(g.GL_FRAMEBUFFER,fbo)
                for label,attachment in [('color',g.GL_COLOR_ATTACHMENT0),('depth',g.GL_DEPTH_ATTACHMENT)]:
                    resource=int(g.glGenRenderbuffers(1));buffers.append(resource);g.glBindRenderbuffer(g.GL_RENDERBUFFER,resource)
                    fmt=descriptor['attachments'][label]['format']
                    if samples:g.glRenderbufferStorageMultisample(g.GL_RENDERBUFFER,samples,fmt,size,size)
                    else:g.glRenderbufferStorage(g.GL_RENDERBUFFER,fmt,size,size)
                    g.glFramebufferRenderbuffer(g.GL_FRAMEBUFFER,attachment,g.GL_RENDERBUFFER,resource)
                g.glDrawBuffer(g.GL_COLOR_ATTACHMENT0);g.glReadBuffer(g.GL_COLOR_ATTACHMENT0)
                if g.glCheckFramebufferStatus(g.GL_FRAMEBUFFER)!=g.GL_FRAMEBUFFER_COMPLETE:raise ValueError('incomplete_scratch_framebuffer')
                if index==0:
                    actual=int(g.glGetIntegerv(g.GL_SAMPLES))
                    positions=[np.asarray(g.glGetMultisamplefv(g.GL_SAMPLE_POSITION,i),float).tolist() for i in range(actual)]
                    if actual!=n or not np.allclose(positions,descriptor['sample_positions_bottom_origin_xy'],rtol=0,atol=1e-8):raise ValueError('scratch_sample_positions_differ')
            g.glViewport(0,0,size,size);g.glDepthRange(0,1)
            for cap in (g.GL_CULL_FACE,g.GL_LIGHTING,g.GL_FOG,g.GL_BLEND,g.GL_SCISSOR_TEST,g.GL_ALPHA_TEST,g.GL_STENCIL_TEST,g.GL_POLYGON_OFFSET_FILL,g.GL_RASTERIZER_DISCARD,g.GL_SAMPLE_ALPHA_TO_COVERAGE,g.GL_SAMPLE_ALPHA_TO_ONE,g.GL_SAMPLE_COVERAGE):g.glDisable(cap)
            for i in range(int(g.glGetIntegerv(g.GL_MAX_CLIP_PLANES))):g.glDisable(g.GL_CLIP_PLANE0+i)
            g.glPolygonMode(g.GL_FRONT_AND_BACK,g.GL_FILL)
            g.glEnable(g.GL_DEPTH_TEST);g.glDepthFunc(g.GL_ALWAYS);g.glDepthMask(g.GL_TRUE);g.glColorMask(True,True,True,True)
            g.glEnable(g.GL_MULTISAMPLE);g.glEnable(g.GL_SAMPLE_MASK)
            trials=[];base=np.linspace(.15,.85,n)
            for shift in range(n):
                depths=np.roll(base,shift)
                g.glBindFramebuffer(g.GL_FRAMEBUFFER,fbos[0]);g.glSampleMaski(0,0xffffffff);g.glClearDepth(1.);g.glClear(g.GL_DEPTH_BUFFER_BIT|g.GL_COLOR_BUFFER_BIT)
                for i,depth in enumerate(depths):
                    g.glSampleMaski(0,1<<i);g.glBegin(g.GL_QUADS)
                    z=depth if clip_mode=='GL_ZERO_TO_ONE' else 2*depth-1
                    for x,y in ((-1,-1),(1,-1),(1,1),(-1,1)):g.glVertex3f(x,y,z)
                    g.glEnd()
                g.glSampleMaski(0,0xffffffff)
                g.glBindFramebuffer(g.GL_READ_FRAMEBUFFER,fbos[0]);g.glBindFramebuffer(g.GL_DRAW_FRAMEBUFFER,fbos[1])
                g.glBlitFramebuffer(0,0,size,size,0,0,size,size,g.GL_COLOR_BUFFER_BIT|g.GL_DEPTH_BUFFER_BIT,g.GL_NEAREST)
                g.glBindFramebuffer(g.GL_READ_FRAMEBUFFER,fbos[1])
                values=[float(np.asarray(g.glReadPixels(x,y,1,1,g.GL_DEPTH_COMPONENT,g.GL_FLOAT)).ravel()[0]) for x,y in ((1,1),(2,3),(5,6))]
                trials.append(dict(assigned_sample_depths=depths.tolist(),resolved_depths=values))
            return positions,trials
        finally:
            # Delete only our own resources, then restore bindings and every
            # modified compatibility GL attribute/matrix. Live task pixels were
            # never a draw/blit destination.
            if fbos:g.glDeleteFramebuffers(len(fbos),fbos)
            if buffers:g.glDeleteRenderbuffers(len(buffers),buffers)
            # Attribute restore includes draw/read buffer selection; restore
            # its original framebuffer before PopAttrib (COLOR_ATTACHMENT0 is
            # invalid on the default framebuffer after deleting scratch FBOs).
            g.glBindFramebuffer(g.GL_READ_FRAMEBUFFER,read);g.glBindFramebuffer(g.GL_DRAW_FRAMEBUFFER,draw)
            for mode in reversed(matrices):g.glMatrixMode(mode);g.glPopMatrix()
            if pushed:g.glPopAttrib()
            g.glMatrixMode(matrix_mode);g.glUseProgram(program)
            for i,value in enumerate(masks):g.glSampleMaski(i,value & 0xffffffff)
            # GL_SAMPLE_MASK is an extension enable, not restored by the
            # compatibility GL_ALL_ATTRIB_BITS stack on all implementations.
            (g.glEnable if sample_mask_enabled else g.glDisable)(g.GL_SAMPLE_MASK)
            g.glBindRenderbuffer(g.GL_RENDERBUFFER,rb)
            g.glBindFramebuffer(g.GL_READ_FRAMEBUFFER,read);g.glBindFramebuffer(g.GL_DRAW_FRAMEBUFFER,draw)
