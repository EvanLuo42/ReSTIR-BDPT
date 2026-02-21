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
#include "ReSTIRBDPTPass.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "Utils/Sampling/AliasTable.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIRBDPTPass>();
}

ReSTIRBDPTPass::ReSTIRBDPTPass(ref<Device> pDevice, const Properties& props) : RenderPass(std::move(pDevice)) {}

Properties ReSTIRBDPTPass::getProperties() const
{
    return {};
}

void ReSTIRBDPTPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) {}

void ReSTIRBDPTPass::GenerateAliasTable(const ref<Scene>& pScene)
{
    std::vector<float> weights;

    for (uint32_t i = 0; i < pScene->getLightCount(); ++i)
    {
        auto light = pScene->getLight(i);
        weights.push_back(light->getPower());
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    mpAliasTable = std::make_unique<AliasTable>(mpDevice, weights, rng);
}

RenderPassReflection ReSTIRBDPTPass::reflect(const CompileData& compileData)
{
    RenderPassReflection r;

    r.addOutput("color", "Final output");

    return r;
}

void ReSTIRBDPTPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mFrameDim = compileData.defaultTexDims;
    mNumLightSubpaths = mFrameDim.x * mFrameDim.y;
}

void ReSTIRBDPTPass::execute(RenderContext* pRenderContext, const RenderData& renderData) {}

void ReSTIRBDPTPass::renderUI(Gui::Widgets& widget) {}
