from falcor import *

def render_graph_ReSTIRBDPT():
    g = RenderGraph("ReSTIRBDPT")

    ReSTIRBDPT = createPass("ReSTIRBDPTPass", {})
    g.addPass(ReSTIRBDPT, "ReSTIRBDPTPass")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("ReSTIRBDPTPass.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")

    return g

ReSTIRBDPT = render_graph_ReSTIRBDPT()
try: m.addGraph(ReSTIRBDPT)
except NameError: None
