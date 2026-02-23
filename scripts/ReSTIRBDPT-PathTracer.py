from falcor import *


def render_graph_ReSTIRBDPT_PathTracer():
    g = RenderGraph("ReSTIR BDPT vs Path Tracer")
    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1})
    g.addPass(PathTracer, "PathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePassPT = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePassPT, "AccumulatePassPT")
    ToneMapperPT = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapperPT, "ToneMapperPT")
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
    g.addEdge("PathTracer.color", "AccumulatePassPT.input")
    g.addEdge("AccumulatePassPT.output", "ToneMapperPT.src")

    ReSTIRBDPT = createPass("ReSTIRBDPTPass", {})
    g.addPass(ReSTIRBDPT, "ReSTIRBDPTPass")
    AccumulatePassBDPT = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePassBDPT, "AccumulatePassBDPT")
    ToneMapperBDPT = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapperBDPT, "ToneMapperBDPT")

    g.addEdge("ReSTIRBDPTPass.color", "AccumulatePassBDPT.input")
    g.addEdge("AccumulatePassBDPT.output", "ToneMapperBDPT.src")

    SideBySide = createPass("SideBySidePass", {
        'leftLabel': 'ReSTIR BDPT',
        'rightLabel': 'Path Tracer',
        'showTextLabels': True
    })
    g.addPass(SideBySide, "SideBySide")
    g.addEdge("ToneMapperBDPT.dst", "SideBySide.leftInput")
    g.addEdge("ToneMapperPT.dst", "SideBySide.rightInput")

    g.markOutput("SideBySide.output")

    return g


Graph = render_graph_ReSTIRBDPT_PathTracer()
try:
    m.addGraph(Graph)
except NameError:
    None
