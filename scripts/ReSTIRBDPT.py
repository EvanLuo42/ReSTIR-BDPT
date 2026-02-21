from falcor import *

def render_graph_ReSTIRBDPT():
    g = RenderGraph("ReSTIRBDPT")

    ReSTIRBDPT = createPass("ReSTIRBDPTPass", {})
    g.addPass(ReSTIRBDPT, "ReSTIRBDPTPass")

    g.markOutput("ReSTIRBDPTPass.color")

    return g

ReSTIRBDPT = render_graph_ReSTIRBDPT()
try: m.addGraph(ReSTIRBDPT)
except NameError: None
