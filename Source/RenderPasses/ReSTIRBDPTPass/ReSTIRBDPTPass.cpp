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

#include <memory>
#include <utility>
#include "Core/API/Buffer.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Utils/Logger.h"
#include "Utils/Properties.h"
#include "RenderGraph/RenderPassStandardFlags.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIRBDPTPass>();
}

namespace
{
const char kGenerateLightSubpathsShader[] = "RenderPasses/ReSTIRBDPTPass/GenerateLightSubpaths.cs.slang";
const char kCameraTraceAndConnectShader[] = "RenderPasses/ReSTIRBDPTPass/CameraTraceAndConnect.cs.slang";
const char kFinalResolveShader[] = "RenderPasses/ReSTIRBDPTPass/FinalResolve.cs.slang";

const float kRoughnessThreshold = 0.08f;

const char kEnableSpatialReuse[] = "enableSpatialReuse";
const char kEnableTemperoralReuse[] = "enableTemperoalReuse";
const char kNumMaxBounces[] = "numMaxBounces";
} // namespace

ReSTIRBDPTPass::ReSTIRBDPTPass(ref<Device> pDevice, const Properties& props) : RenderPass(std::move(pDevice))
{
    parseProperties(props);
}

void ReSTIRBDPTPass::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kEnableSpatialReuse)
            mEnableSpatialReuse = value;
        else if (key == kEnableTemperoralReuse)
            mEnableTemporalReuse = value;
        else if (key == kNumMaxBounces)
            mNumMaxBounces = value;
        else
            logWarning("Unknown property '{}' in ReSTIRBDPTPass properties.", key);
    }
}

Properties ReSTIRBDPTPass::getProperties() const
{
    Properties props;
    props[kEnableSpatialReuse] = mEnableSpatialReuse;
    props[kEnableTemperoralReuse] = mEnableTemporalReuse;
    props[kNumMaxBounces] = mNumMaxBounces;
    return props;
}

void ReSTIRBDPTPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mFrameCount = 0;

    mpScene = pScene;

    if (!mpScene)
        return;

    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_TINY_UNIFORM);
    mpEmissivePowerSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext));

    DefineList defines = mpScene->getSceneDefines();
    defines.add(mpSampleGenerator->getDefines());
    defines.add(mpEmissivePowerSampler->getDefines());

    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.addShaderLibrary(kGenerateLightSubpathsShader).csEntry("main");
        mpGenerateLightSubpathsPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.addShaderLibrary(kCameraTraceAndConnectShader).csEntry("main");
        mpCameraTraceAndConnectPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addTypeConformances(mpScene->getTypeConformances());
        desc.addShaderLibrary(kFinalResolveShader).csEntry("main");
        mpFinalResolvePass = ComputePass::create(mpDevice, desc, defines, true);
    }

    uint32_t maxLVCSize = mNumLightSubpaths * mNumMaxBounces;

    auto lightVar = mpGenerateLightSubpathsPass->getRootVar();

    mpLVC = mpDevice->createStructuredBuffer(
        lightVar["gLVC"],
        maxLVCSize,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr,
        true
    );

    uint32_t numPixels = mNumLightSubpaths;

    auto cameraVar = mpCameraTraceAndConnectPass->getRootVar();
    mpOutputReservoirs = mpDevice->createStructuredBuffer(
        cameraVar["gOutputReservoirs"],
        numPixels,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr,
        false
    );
    mpOutputCausticReservoirs = mpDevice->createStructuredBuffer(
        cameraVar["gOutputCausticReservoirs"],
        numPixels,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr,
        false
    );

    if (mFrameDim.x > 0 && mFrameDim.y > 0)
    {
        auto var = mpCameraTraceAndConnectPass->getRootVar();
        mpLRM = LightReservoirMap::create(mpDevice, mpScene, var["gLRM"], mFrameDim, mNumLightSubpaths);
    }

    mpDebugCounter = mpDevice->createStructuredBuffer(
        sizeof(uint32_t), 1, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false
    );
}

RenderPassReflection ReSTIRBDPTPass::reflect(const CompileData& compileData)
{
    RenderPassReflection r;

    r.addOutput("color", "Final output")
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::RGBA32Float);

    return r;
}

void ReSTIRBDPTPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mFrameDim = compileData.defaultTexDims;
    mNumLightSubpaths = mFrameDim.x * mFrameDim.y;
}

void ReSTIRBDPTPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[kRenderPassRefreshFlags] = flags | RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    if (!mpScene)
    {
        return;
    }

    mpEmissivePowerSampler->update(pRenderContext, mpScene->getILightCollection(pRenderContext));
    mpLRM->beginFrame(pRenderContext);

    pRenderContext->clearUAVCounter(mpLVC, 0);

    pRenderContext->clearUAV(mpDebugCounter->getUAV().get(), uint4(0));

    auto varLight = mpGenerateLightSubpathsPass->getRootVar();

    mpSampleGenerator->bindShaderData(varLight);
    mpEmissivePowerSampler->bindShaderData(varLight["gEmissivePowerSampler"]);
    mpScene->bindShaderDataForRaytracing(pRenderContext, varLight["gScene"]);

    varLight["CB"]["gNumLightSubpaths"] = mNumLightSubpaths;
    varLight["CB"]["gNumBounces"] = mNumMaxBounces;
    varLight["CB"]["gMISPower"] = 1.0f;
    varLight["CB"]["gFrameCount"] = mFrameCount;
    varLight["CB"]["gTargetDim"] = mFrameDim;
    varLight["CB"]["gRoughnessThreshold"] = kRoughnessThreshold;

    varLight["gLVC"] = mpLVC;
    mpLRM->bindShaderData(varLight["gLRM"]);

    varLight["gDebugCounter"] = mpDebugCounter;

    mpGenerateLightSubpathsPass->execute(pRenderContext, mFrameDim.x * mFrameDim.y, 1, 1);

    mpLRM->executeSort(pRenderContext);

    auto varCamera = mpCameraTraceAndConnectPass->getRootVar();

    mpSampleGenerator->bindShaderData(varCamera);
    mpEmissivePowerSampler->bindShaderData(varCamera["gEmissivePowerSampler"]);
    mpScene->bindShaderDataForRaytracing(pRenderContext, varCamera["gScene"]);

    varCamera["CB"]["gNumLightSubpaths"] = mNumLightSubpaths;
    varCamera["CB"]["gNumBounces"] = mNumMaxBounces;
    varCamera["CB"]["gMISPower"] = 1.0f;
    varCamera["CB"]["gFrameCount"] = mFrameCount;
    varCamera["CB"]["gTargetDim"] = mFrameDim;
    varCamera["CB"]["gRoughnessThreshold"] = kRoughnessThreshold;

    varCamera["gLVC"] = mpLVC;
    mpLRM->bindShaderData(varCamera["gLRM"]);
    varCamera["gOutputReservoirs"] = mpOutputReservoirs;
    varCamera["gOutputCausticReservoirs"] = mpOutputCausticReservoirs;

    mpCameraTraceAndConnectPass->execute(pRenderContext, uint3(mFrameDim, 1));

    auto varResolve = mpFinalResolvePass->getRootVar();

    auto pDstColor = renderData.getTexture("color");
    if (pDstColor)
    {
        varResolve["gOutputReservoirs"] = mpOutputReservoirs;
        varResolve["gOutputCausticReservoirs"] = mpOutputCausticReservoirs;

        varResolve["gOutputColor"] = pDstColor;

        varResolve["CB"]["gTargetDim"] = mFrameDim;

        pRenderContext->clearUAV(pDstColor->getUAV().get(), uint4(0x3f800000, 0, 0, 0x3f800000));
        mpFinalResolvePass->execute(pRenderContext, uint3(mFrameDim, 1));
    }

    mFrameCount++;
}

void ReSTIRBDPTPass::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.var("Max bounces", mNumMaxBounces, 0u, 1u << 16);
    widget.tooltip("Maximum number of bounces when tracing new vertex.", true);

    dirty |= widget.checkbox("Enable Spatial Reuse", mEnableSpatialReuse);
    widget.tooltip("Enable spatial reuse.", true);

    dirty |= widget.checkbox("Enable Temperal Reuse", mEnableTemporalReuse);
    widget.tooltip("Enable temperal reuse", true);

    if (dirty)
    {
        mOptionsChanged = true;
    }
}
