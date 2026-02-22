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
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Utils/Logger.h"
#include "Utils/Properties.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Timing/Profiler.h"

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
    mpScene = pScene;

    mpGenerateLightSubpathsPass = nullptr;
    mpCameraTraceAndConnectPass = nullptr;
    mpFinalResolvePass = nullptr;
    mpLVC = nullptr;
    mpOutputReservoirs = nullptr;
    mpOutputCausticReservoirs = nullptr;
    mpDebugCounter = nullptr;
    mpLRM = nullptr;
    mpSampleGenerator = nullptr;
    mpEmissivePowerSampler = nullptr;

    mFrameCount = 0;
    mRecompile = true;

    if (!mpScene)
        return;

    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_TINY_UNIFORM);
    mpEmissivePowerSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext));
}

void ReSTIRBDPTPass::preparePasses()
{
    FALCOR_ASSERT(mpScene);

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

    mReallocate = true;
}

void ReSTIRBDPTPass::prepareResources()
{
    FALCOR_ASSERT(mpGenerateLightSubpathsPass && mpCameraTraceAndConnectPass);
    FALCOR_ASSERT(mFrameDim.x > 0 && mFrameDim.y > 0);

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

    auto cameraVar = mpCameraTraceAndConnectPass->getRootVar();
    mpOutputReservoirs = mpDevice->createStructuredBuffer(
        cameraVar["gOutputReservoirs"],
        mNumLightSubpaths,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr,
        false
    );
    mpOutputCausticReservoirs = mpDevice->createStructuredBuffer(
        cameraVar["gOutputCausticReservoirs"],
        mNumLightSubpaths,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr,
        false
    );

    mpLRM = LightReservoirMap::create(mpDevice, mpScene, cameraVar["gLRM"], mFrameDim, mNumLightSubpaths);

    mpDebugCounter = mpDevice->createStructuredBuffer(
        sizeof(uint32_t), 1, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false
    );

    mFrameCount = 0;
}

void ReSTIRBDPTPass::executeGenerateLightSubpaths(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "GenerateLightSubpaths");

    pRenderContext->clearUAVCounter(mpLVC, 0);
    pRenderContext->clearUAV(mpDebugCounter->getUAV().get(), uint4(0));

    auto var = mpGenerateLightSubpathsPass->getRootVar();

    mpSampleGenerator->bindShaderData(var);
    mpEmissivePowerSampler->bindShaderData(var["gEmissivePowerSampler"]);
    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

    var["CB"]["gNumLightSubpaths"] = mNumLightSubpaths;
    var["CB"]["gNumBounces"] = mNumMaxBounces;
    var["CB"]["gMISPower"] = 1.0f;
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gTargetDim"] = mFrameDim;
    var["CB"]["gRoughnessThreshold"] = kRoughnessThreshold;

    var["gLVC"] = mpLVC;
    mpLRM->bindShaderData(var["gLRM"]);
    var["gDebugCounter"] = mpDebugCounter;

    mpGenerateLightSubpathsPass->execute(pRenderContext, mNumLightSubpaths, 1, 1);
}

void ReSTIRBDPTPass::executeCameraTraceAndConnect(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "CameraTraceAndConnect");

    auto var = mpCameraTraceAndConnectPass->getRootVar();

    mpSampleGenerator->bindShaderData(var);
    mpEmissivePowerSampler->bindShaderData(var["gEmissivePowerSampler"]);
    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

    var["CB"]["gNumLightSubpaths"] = mNumLightSubpaths;
    var["CB"]["gNumBounces"] = mNumMaxBounces;
    var["CB"]["gMISPower"] = 1.0f;
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gTargetDim"] = mFrameDim;
    var["CB"]["gRoughnessThreshold"] = kRoughnessThreshold;

    var["gLVC"] = mpLVC;
    mpLRM->bindShaderData(var["gLRM"]);
    var["gOutputReservoirs"] = mpOutputReservoirs;
    var["gOutputCausticReservoirs"] = mpOutputCausticReservoirs;

    mpCameraTraceAndConnectPass->execute(pRenderContext, uint3(mFrameDim, 1));
}

void ReSTIRBDPTPass::executeFinalResolve(RenderContext* pRenderContext, const ref<Texture>& pDstColor)
{
    FALCOR_PROFILE(pRenderContext, "FinalResolve");

    pRenderContext->clearUAV(pDstColor->getUAV().get(), uint4(0x3f800000, 0, 0, 0x3f800000));

    auto var = mpFinalResolvePass->getRootVar();
    var["gOutputReservoirs"] = mpOutputReservoirs;
    var["gOutputCausticReservoirs"] = mpOutputCausticReservoirs;
    var["gOutputColor"] = pDstColor;
    var["CB"]["gTargetDim"] = mFrameDim;

    mpFinalResolvePass->execute(pRenderContext, uint3(mFrameDim, 1));
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
    const uint2 newDim = compileData.defaultTexDims;
    if (newDim.x != mFrameDim.x || newDim.y != mFrameDim.y)
    {
        mFrameDim.x = newDim.x;
        mFrameDim.y = newDim.y;
        mNumLightSubpaths = mFrameDim.x * mFrameDim.y;
        mReallocate = true;
    }
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
        return;

    if (mRecompile)
    {
        preparePasses();
        mRecompile = false;
    }

    if (mReallocate)
    {
        if (mFrameDim.x == 0 || mFrameDim.y == 0)
            return;
        prepareResources();
        mReallocate = false;
    }

    mpEmissivePowerSampler->update(pRenderContext, mpScene->getILightCollection(pRenderContext));
    mpLRM->beginFrame(pRenderContext);

    executeGenerateLightSubpaths(pRenderContext);

    {
        FALCOR_PROFILE(pRenderContext, "LRM Sort");
        mpLRM->executeSort(pRenderContext);
    }

    executeCameraTraceAndConnect(pRenderContext);

    auto pDstColor = renderData.getTexture("color");
    if (pDstColor)
        executeFinalResolve(pRenderContext, pDstColor);

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
