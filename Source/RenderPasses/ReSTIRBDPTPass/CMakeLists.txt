add_plugin(ReSTIRBDPTPass)

target_sources(ReSTIRBDPTPass PRIVATE
    ReSTIRBDPTPass.cpp
    ReSTIRBDPTPass.h
    GenerateLightSubpaths.cs.slang
    PathReservoir.slang
    LightReservoirMap.cpp
    LightReservoirMap.h
    LRM.slang
    LRM_Scatter.cs.slang
    CameraTraceAndConnect.cs.slang
    FinalResolve.cs.slang
)

target_copy_shaders(ReSTIRBDPTPass RenderPasses/ReSTIRBDPTPass)

target_source_group(ReSTIRBDPTPass "RenderPasses")
