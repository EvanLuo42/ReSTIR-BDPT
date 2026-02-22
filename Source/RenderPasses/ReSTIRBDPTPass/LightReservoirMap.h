#pragma once
#include <memory>
#include "Core/API/Buffer.h"
#include "Core/API/Device.h"
#include "Core/Pass/ComputePass.h"
#include "Scene/Scene.h"
#include "Utils/Algorithm/PrefixSum.h"

using namespace Falcor;

class LightReservoirMap
{
public:
    LightReservoirMap(ref<Device> pDevice, ref<Scene> pScene, const ShaderVar& var, uint2 resolution, uint maxLightPaths);

    static std::unique_ptr<LightReservoirMap> create(ref<Device> pDevice, ref<Scene> pScene, const ShaderVar& var, uint2 resolution, uint maxLightPaths);

    void beginFrame(RenderContext* pRenderContext);

    void executeSort(RenderContext* pRenderContext);

    void bindShaderData(const ShaderVar& var);

private:
    ref<Device> mpDevice;
    ref<Scene> mpScene;
    uint2 mResolution;
    uint mMaxLightPaths;
    uint mNumPixels;

    ref<Buffer> mpGlobalReservoirs;
    ref<Buffer> mpCellCounters;
    ref<Buffer> mpAppendBuffer;
    ref<Buffer> mpAppendCounter;
    ref<Buffer> mpIndexBuffer;
    ref<Buffer> mpCellStorage;

    ref<SampleGenerator> mpSampleGenerator;

    ref<ComputePass> mpScatterPass;
    std::unique_ptr<PrefixSum> mpPrefixSum;
};