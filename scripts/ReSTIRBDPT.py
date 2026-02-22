from falcor import *

def render_graph_ReSTIRBDPT():
    g = RenderGraph("ReSTIRBDPT")

    ReSTIRBDPT = createPass("ReSTIRBDPTPass", {})
    g.addPass(ReSTIRBDPT, "ReSTIRBDPTPass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("ReSTIRBDPTPass.color", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")

    return g

ReSTIRBDPT = render_graph_ReSTIRBDPT()
try: m.addGraph(ReSTIRBDPT)
except NameError: None
