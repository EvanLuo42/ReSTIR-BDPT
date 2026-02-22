/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once

#include <memory>
#include "Core/API/Buffer.h"
#include "Core/Object.h"
#include "Core/Pass/ComputePass.h"
#include "LightReservoirMap.h"
#include "RenderGraph/RenderPass.h"
#include "Rendering/Lights/EmissiveLightSampler.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Scene/Scene.h"
#include "Utils/Math/VectorTypes.h"
#include "Utils/Properties.h"

using namespace Falcor;

class ReSTIRBDPTPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ReSTIRBDPTPass, "ReSTIRBDPTPass", "Insert pass description here.");

    static ref<ReSTIRBDPTPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<ReSTIRBDPTPass>(pDevice, props); }

    ReSTIRBDPTPass(ref<Device> pDevice, const Properties& props);

    Properties getProperties() const override;
    RenderPassReflection reflect(const CompileData& compileData) override;
    void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    void parseProperties(const Properties& props);

    ref<Scene> mpScene;
    ref<SampleGenerator> mpSampleGenerator;
    std::unique_ptr<EmissivePowerSampler> mpEmissivePowerSampler;

    std::unique_ptr<LightReservoirMap> mpLRM;

    bool mEnableTemporalReuse = true;
    bool mEnableSpatialReuse = true;
    uint mNumMaxBounces = 10;

    int mFrameCount = 0;
    uint2 mFrameDim;

    bool mOptionsChanged = false;

    uint mNumLightSubpaths;

    ref<ComputePass> mpGenerateLightSubpathsPass;
    ref<ComputePass> mpCameraTraceAndConnectPass;
    ref<ComputePass> mpTemporalReusePass;
    ref<ComputePass> mpSpatialReusePass;
    ref<ComputePass> mpFinalResolvePass;

    ref<Buffer> mpLVC;
    ref<Buffer> mpOutputReservoirs;
    ref<Buffer> mpOutputCausticReservoirs;

    ref<Buffer> mpDebugCounter;
};
