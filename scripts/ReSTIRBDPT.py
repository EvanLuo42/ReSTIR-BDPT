from falcor import *

def render_graph_ReSTIRBDPT():
    g = RenderGraph("ReSTIRBDPT")

    ReSTIRBDPT = createPass("ReSTIRBDPTPass", {
        'maxBounces':          8,
        'numLightSubpaths':    1 << 15,   # 32768
        'cellSize':            0.5,
        'lvcCandidates':       1,
        'neeCandidates':       1,
        'neighborCount':       8,
        'enableTemporalReuse': True,
        'enableSpatialReuse':  True,
        'enableCaustics':      True,
        'depthThreshold':      0.01,
        'normalThreshold':     0.9,
        'exposure':            1.0,
        'clamp':               0.0,
    })
    g.addPass(ReSTIRBDPT, "ReSTIRBDPTPass")

    # The pass outputs a tonemapped RGBA32Float image directly.
    g.markOutput("ReSTIRBDPTPass.color")

    return g

ReSTIRBDPT = render_graph_ReSTIRBDPT()
try: m.addGraph(ReSTIRBDPT)
except NameError: None
