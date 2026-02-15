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
#include "Core/API/Buffer.h"
#include "Core/Object.h"
#include "Core/Pass/ComputePass.h"
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Scene/Scene.h"
#include "Utils/Algorithm/PrefixSum.h"

using namespace Falcor;

class ReSTIRBDPTPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ReSTIRBDPTPass, "ReSTIRBDPTPass", "Insert pass description here.");

    static ref<ReSTIRBDPTPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<ReSTIRBDPTPass>(pDevice, props); }

    ReSTIRBDPTPass(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override {}
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    struct StaticParams
    {};

    StaticParams mStaticParams;

    ref<Scene> mpScene;
    ref<SampleGenerator> mpSampleGenerator;

    bool mEnableTemporalReuse = true;
    bool mEnableSpatialReuse = true;

    int mFrameCount = 0;

    int mNumLightSubpaths = 5;

    ref<ComputePass> mpLRMClearPass;
    ref<ComputePass> mpGenerateLightSubpathsPass;
    ref<ComputePass> mpLRMScatterPass;
    ref<ComputePass> mpCameraTraceAndConnectPass;
    ref<ComputePass> mpSpatialReusePass;
    ref<ComputePass> mpTemporalReusePass;
    ref<ComputePass> mpFinalResolvePass;

    static constexpr uint32_t kLRMNumBuckets = 100000;
    static constexpr uint32_t kLRMBucketEntries = 32;
    static constexpr uint32_t kLRMCellEntryCount = kLRMNumBuckets * kLRMBucketEntries;

    static constexpr uint32_t kLRMMaxRecords = 4'000'000;
    static constexpr uint32_t kLRMMaxTriplets = 4'000'000;
    static constexpr uint32_t kLRMMaxCellStorage = 4'000'000;
    static constexpr uint32_t kLRMMaxPerCell = 64;

    // LRM
    ref<Buffer> mpLRMRecords;
    ref<Buffer> mpLRMTriplets;

    ref<Buffer> mpLRMRecordCount;
    ref<Buffer> mpLRMTripletCount;

    ref<Buffer> mpLRMCellChecksum;
    ref<Buffer> mpLRMCellCount;
    ref<Buffer> mpLRMCellOffset;

    ref<Buffer> mpLRMCellStorage;

    // LVC
    ref<Buffer> mpLightVertexCache;

    ref<PrefixSum> mpPrefixSum;

    ref<Buffer> mpReservoir;
    ref<Buffer> mpCausticReservoir;
};
