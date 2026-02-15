add_plugin(ReSTIRBDPTPass)

target_sources(ReSTIRBDPTPass PRIVATE
    ReSTIRBDPTPass.cpp
    ReSTIRBDPTPass.h
)

target_source_group(ReSTIRBDPTPass "RenderPasses")
