add_plugin(ReSTIRBDPTPass)

target_sources(ReSTIRBDPTPass PRIVATE
    ReSTIRBDPTPass.cpp
    ReSTIRBDPTPass.h
    CameraTraceAndConnect.cs.slang
    GenerateLightSubpaths.cs.slang
    LightUtils.slang
    LRM.slangh
    LRMClear.cs.slang
    LRMScatter.cs.slang
    PathReservoir.slangg
)

target_copy_shaders(ReSTIRBDPTPass RenderPasses/ReSTIRBDPTPass)

target_source_group(ReSTIRBDPTPass "RenderPasses")
