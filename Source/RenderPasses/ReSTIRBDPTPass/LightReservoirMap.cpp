#include "LightReservoirMap.h"
#include <cstddef>
#include <memory>
#include "Core/API/Buffer.h"
#include "Core/API/Device.h"
#include "Core/Program/ShaderVar.h"
#include "Scene/Scene.h"
#include "Utils/Algorithm/PrefixSum.h"

std::unique_ptr<LightReservoirMap> LightReservoirMap::create(ref<Device> pDevice, ref<Scene> pScene, const ShaderVar& var, uint2 resolution, uint maxLightPaths)
{
    return std::make_unique<LightReservoirMap>(pDevice, pScene, var, resolution, maxLightPaths);
}

LightReservoirMap::LightReservoirMap(ref<Device> pDevice, ref<Scene> pScene, const ShaderVar& var, uint2 resolution, uint maxLightPaths)
    : mpDevice(pDevice), mpScene(pScene), mResolution(resolution), mMaxLightPaths(maxLightPaths)
{
    mNumPixels = resolution.x * resolution.y;

    mpGlobalReservoirs = pDevice->createStructuredBuffer(
        var["globalReservoirs"],
        mMaxLightPaths,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr,
        false
    );

    mpCellCounters = pDevice->createStructuredBuffer(
        sizeof(uint32_t),
        mNumPixels,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr,
        false
    );

    mpAppendBuffer = pDevice->createStructuredBuffer(
        sizeof(uint32_t) * 3,
        mMaxLightPaths,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr,
        false
    );

    mpAppendCounter = pDevice->createStructuredBuffer(
        sizeof(uint32_t), 1, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false
    );

    mpIndexBuffer = pDevice->createStructuredBuffer(
        sizeof(uint32_t),
        mNumPixels,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr,
        false
    );

    mpCellStorage = pDevice->createStructuredBuffer(
        sizeof(uint32_t),
        mMaxLightPaths,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr,
        false
    );

    mpPrefixSum = std::make_unique<PrefixSum>(pDevice);
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_TINY_UNIFORM);

    DefineList defines = mpScene->getSceneDefines();
    defines.add(mpSampleGenerator->getDefines());
    mpScatterPass = ComputePass::create(mpDevice, "RenderPasses/ReSTIRBDPTPass/LRM_Scatter.cs.slang", "main", defines);
}

void LightReservoirMap::beginFrame(RenderContext* pRenderContext)
{
    pRenderContext->clearUAV(mpAppendCounter->getUAV().get(), uint4(0));

    pRenderContext->clearUAV(mpCellCounters->getUAV().get(), uint4(0));
}

void LightReservoirMap::executeSort(RenderContext* pRenderContext)
{
    pRenderContext->copyResource(mpIndexBuffer.get(), mpCellCounters.get());

    mpPrefixSum->execute(pRenderContext, mpIndexBuffer, mNumPixels);

    auto var = mpScatterPass->getRootVar()["gLRM"];
    bindShaderData(var);

    uint32_t threadGroupCount = (mMaxLightPaths + 255) / 256;
    mpScatterPass->execute(pRenderContext, threadGroupCount, 1, 1);
}

void LightReservoirMap::bindShaderData(const ShaderVar& var)
{
    var["globalReservoirs"] = mpGlobalReservoirs;
    var["cellCounters"] = mpCellCounters;
    var["appendBuffer"] = mpAppendBuffer;
    var["appendCounter"] = mpAppendCounter;
    var["indexBuffer"] = mpIndexBuffer;
    var["cellStorage"] = mpCellStorage;
    var["maxLightPaths"] = mMaxLightPaths;
    var["numPixels"] = mNumPixels;
}