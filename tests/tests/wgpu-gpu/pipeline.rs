use wgpu_test::{
    fail, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE,
        COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX,
        RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE,
        RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX,
        NO_TARGETLESS_RENDER,
        RENDER_PIPELINE_ASYNC_RESOLVES_A_USABLE_PIPELINE,
        RENDER_PIPELINE_ASYNC_REPORTS_AN_INVALID_DESCRIPTOR_THROUGH_THE_FUTURE,
        COMPUTE_PIPELINE_ASYNC_REPORTS_AN_INVALID_DESCRIPTOR_THROUGH_THE_FUTURE,
        RENDER_PIPELINES_ASYNC_ALL_RESOLVE_TO_USABLE_PIPELINES,
        RENDER_PIPELINE_SYNC_SHARING_A_PROGRAM_STILL_LINKING_IS_USABLE_AT_ONCE,
        RENDER_PIPELINE_ASYNC_DROPPED_BEFORE_IT_RESOLVES_LEAVES_THE_DEVICE_USABLE,
        RENDER_PIPELINE_ASYNC_ON_A_DESTROYED_DEVICE_RESOLVES,
        RENDER_PIPELINE_ASYNC_REPORTS_A_FAILED_COMPILE_THROUGH_THE_FUTURE,
        RENDER_PIPELINE_SYNC_REPORTS_A_FAILED_COMPILE_THROUGH_THE_ERROR_SCOPE,
    ]);
}

/// Polls the device until `pending` resolves. On GL, a pipeline whose shaders compile in the
/// background resolves on the poll or submission that finds them compiled.
async fn resolved<T>(ctx: &TestingContext, pending: impl core::future::Future<Output = T>) -> T {
    let mut pending = core::pin::pin!(pending);
    loop {
        if let Some(resolved) = futures_lite::future::poll_once(pending.as_mut()).await {
            return resolved;
        }
        ctx.async_poll(wgpu::PollType::Poll).await.unwrap();
    }
}

/// A fragment shader that writes `colour` to its one target, distinct for each `colour`.
fn coloured_fragment_shader(ctx: &TestingContext, colour: u32) -> wgpu::ShaderModule {
    ctx.device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("coloured fragment shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(format!(
                "@fragment fn main() -> @location(0) vec4<f32> {{ return vec4<f32>(f32({colour}) / 255.0, 0.0, 0.0, 1.0); }}"
            ))),
        })
}

/// A pipeline drawing with `vertex` and `fragment` into one RGBA8 target, blending as
/// `blend` says.
fn trivial_pipeline_descriptor<'a>(
    vertex: &'a wgpu::ShaderModule,
    fragment: &'a wgpu::ShaderModule,
    layout: Option<&'a wgpu::PipelineLayout>,
    targets: &'a [Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipelineDescriptor<'a> {
    wgpu::RenderPipelineDescriptor {
        label: Some("trivial render pipeline"),
        layout,
        vertex: wgpu::VertexState {
            module: vertex,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: fragment,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            targets,
        }),
        multiview_mask: None,
        cache: None,
    }
}

/// One RGBA8 target, written without blending, or blended as `blend` says.
fn rgba8_target(blend: Option<wgpu::BlendState>) -> [Option<wgpu::ColorTargetState>; 1] {
    [Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba8Unorm,
        blend,
        write_mask: wgpu::ColorWrites::ALL,
    })]
}

/// Draws one triangle with each of `pipelines` into a 1x1 RGBA8 target and waits for the
/// device to finish.
async fn draw_with(ctx: &TestingContext, pipelines: &[&wgpu::RenderPipeline]) {
    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pipeline test target"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pipeline test pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations::default(),
            })],
            ..Default::default()
        });
        for pipeline in pipelines {
            pass.set_pipeline(pipeline);
            pass.draw(0..3, 0..1);
        }
    }
    ctx.queue.submit([encoder.finish()]);
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
}

/// A GLSL ES fragment shader that cannot compile, given to GL as it is.
const UNCOMPILABLE_GLSL_FRAGMENT: &str =
    "#version 300 es\nprecision highp float;\nout vec4 colour;\nvoid main() { colour = no_such_value; }\n";

/// A GLSL ES vertex shader that compiles, given to GL as it is.
const GLSL_VERTEX: &str =
    "#version 300 es\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";

/// A shader module GL takes as the GLSL `source` given, for a stage with entry point `main`.
fn glsl_module(ctx: &TestingContext, source: &'static str) -> wgpu::ShaderModule {
    unsafe {
        ctx.device
            .create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some("GLSL passthrough module"),
                entry_points: std::borrow::Cow::Borrowed(&[wgpu::PassthroughShaderEntryPoint {
                    name: std::borrow::Cow::Borrowed("main"),
                    workgroup_size: (0, 0, 0),
                }]),
                glsl: Some(std::borrow::Cow::Borrowed(source)),
                ..Default::default()
            })
    }
}

/// A pipeline whose fragment stage cannot compile, from GLSL given to GL as it is.
fn uncompilable_pipeline_parts(
    ctx: &TestingContext,
) -> (wgpu::ShaderModule, wgpu::ShaderModule, wgpu::PipelineLayout) {
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("empty layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
    (
        glsl_module(ctx, GLSL_VERTEX),
        glsl_module(ctx, UNCOMPILABLE_GLSL_FRAGMENT),
        layout,
    )
}

const INVALID_SHADER_DESC: wgpu::ShaderModuleDescriptor = wgpu::ShaderModuleDescriptor {
    label: Some("invalid shader"),
    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed("not valid wgsl")),
};

const TRIVIAL_COMPUTE_SHADER_DESC: wgpu::ShaderModuleDescriptor = wgpu::ShaderModuleDescriptor {
    label: Some("trivial compute shader"),
    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
        "@compute @workgroup_size(1) fn main() {}",
    )),
};

const TRIVIAL_VERTEX_SHADER_DESC: wgpu::ShaderModuleDescriptor = wgpu::ShaderModuleDescriptor {
    label: Some("trivial vertex shader"),
    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
        "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0); }",
    )),
};

const TRIVIAL_FRAGMENT_SHADER_DESC: wgpu::ShaderModuleDescriptor = wgpu::ShaderModuleDescriptor {
    label: Some("trivial fragment shader"),
    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
        "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(0); }",
    )),
};

// Create an invalid shader and a compute pipeline that uses it
// with a default bindgroup layout, and then ask for that layout.
// Validation should fail, but wgpu should not panic.
#[gpu_test]
static COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_sync(|ctx| {
            fail(
                &ctx.device,
                || {
                    let module = ctx.device.create_shader_module(INVALID_SHADER_DESC);

                    let pipeline =
                        ctx.device
                            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                label: Some("compute pipeline"),
                                layout: None,
                                module: &module,
                                entry_point: Some("doesn't exist"),
                                compilation_options: Default::default(),
                                cache: None,
                            });

                    // https://github.com/gfx-rs/wgpu/issues/4167 this used to panic
                    pipeline.get_bind_group_layout(0);
                },
                Some("Shader 'invalid shader' parsing error"),
            );
        });

#[gpu_test]
static COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .enable_noop(),
        )
        .run_sync(|ctx| {
            fail(
                &ctx.device,
                || {
                    let module = ctx.device.create_shader_module(TRIVIAL_COMPUTE_SHADER_DESC);

                    let pipeline =
                        ctx.device
                            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                label: Some("compute pipeline"),
                                layout: None,
                                module: &module,
                                entry_point: Some("main"),
                                compilation_options: Default::default(),
                                cache: None,
                            });

                    pipeline.get_bind_group_layout(u32::MAX);
                },
                Some("Bind group layout index 4294967295 is greater than the device's configured `max_bind_groups` limit"),
            );
        });

#[gpu_test]
static RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_sync(|ctx| {
            fail(
                &ctx.device,
                || {
                    let module = ctx.device.create_shader_module(INVALID_SHADER_DESC);

                    let pipeline =
                        ctx.device
                            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                                label: Some("render pipeline"),
                                layout: None,
                                vertex: wgpu::VertexState {
                                    module: &module,
                                    entry_point: Some("doesn't exist"),
                                    compilation_options: Default::default(),
                                    buffers: &[],
                                },
                                primitive: Default::default(),
                                depth_stencil: None,
                                multisample: Default::default(),
                                fragment: None,
                                multiview_mask: None,
                                cache: None,
                            });

                    pipeline.get_bind_group_layout(0);
                },
                Some("Shader 'invalid shader' parsing error"),
            );
        });

#[gpu_test]
static RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .enable_noop(),
        )
        .run_sync(|ctx| {
            fail(
                &ctx.device,
                || {
                    let vs_module = ctx.device.create_shader_module(TRIVIAL_VERTEX_SHADER_DESC);
                    let fs_module = ctx
                        .device
                        .create_shader_module(TRIVIAL_FRAGMENT_SHADER_DESC);

                    let pipeline =
                        ctx.device
                            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                                label: Some("render pipeline"),
                                layout: None,
                                vertex: wgpu::VertexState {
                                    module: &vs_module,
                                    entry_point: Some("main"),
                                    compilation_options: Default::default(),
                                    buffers: &[],
                                },
                                primitive: Default::default(),
                                depth_stencil: None,
                                multisample: Default::default(),
                                fragment: Some(wgpu::FragmentState {
                                    module: &fs_module,
                                    entry_point: Some("main"),
                                    compilation_options: Default::default(),
                                    targets: &[Some(wgpu::ColorTargetState {
                                        format: wgpu::TextureFormat::Rgba8Unorm,
                                        blend: None,
                                        write_mask: wgpu::ColorWrites::ALL,
                                    })],
                                }),
                                multiview_mask: None,
                                cache: None,
                            });

                    pipeline.get_bind_group_layout(u32::MAX);
                },
                Some("Bind group layout index 4294967295 is greater than the device's configured `max_bind_groups` limit"),
            );
        });

#[gpu_test]
static NO_TARGETLESS_RENDER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        fail(
            &ctx.device,
            || {
                // Testing multisampling is important, because some backends don't behave well if one
                // tries to compile code in an unsupported multisample count. Failing to validate here
                // has historically resulted in requesting the back end to compile code.
                for power_of_two in [1, 2, 4, 8, 16, 32, 64] {
                    let _ = ctx
                        .device
                        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: None,
                            layout: None,
                            vertex: wgpu::VertexState {
                                module: &ctx
                                    .device
                                    .create_shader_module(TRIVIAL_VERTEX_SHADER_DESC),
                                entry_point: Some("main"),
                                compilation_options: Default::default(),
                                buffers: &[],
                            },
                            primitive: Default::default(),
                            depth_stencil: None,
                            multisample: wgpu::MultisampleState {
                                count: power_of_two,
                                ..Default::default()
                            },
                            fragment: None,
                            multiview_mask: None,
                            cache: None,
                        });
                }
            },
            Some(concat!(
                "At least one color attachment or depth-stencil attachment was expected, ",
                "but no render target for the pipeline was specified."
            )),
        )
    });

// A pipeline awaited from `create_render_pipeline_async` must be usable for drawing: on the
// backends without an asynchronous form the future is already resolved, and on WebGPU it
// resolves with the pipeline `createRenderPipelineAsync()` compiled.
#[gpu_test]
static RENDER_PIPELINE_ASYNC_RESOLVES_A_USABLE_PIPELINE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_async(|ctx| async move {
            let vs_module = ctx.device.create_shader_module(TRIVIAL_VERTEX_SHADER_DESC);
            let fs_module = ctx
                .device
                .create_shader_module(TRIVIAL_FRAGMENT_SHADER_DESC);

            let pipeline =
                ctx.device
                    .create_render_pipeline_async(&wgpu::RenderPipelineDescriptor {
                        label: Some("async render pipeline"),
                        layout: None,
                        vertex: wgpu::VertexState {
                            module: &vs_module,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            buffers: &[],
                        },
                        primitive: Default::default(),
                        depth_stencil: None,
                        multisample: Default::default(),
                        fragment: Some(wgpu::FragmentState {
                            module: &fs_module,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        multiview_mask: None,
                        cache: None,
                    });
            let pipeline = resolved(&ctx, pipeline)
                .await
                .expect("async render pipeline creation failed");

            let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("async render pipeline target"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("async render pipeline pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations::default(),
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&pipeline);
                pass.draw(0..3, 0..1);
            }
            ctx.queue.submit([encoder.finish()]);

            ctx.async_poll(wgpu::PollType::wait_indefinitely())
                .await
                .unwrap();
        });

// An invalid descriptor is reported through the future only: WebGPU rejects the promise,
// and wgpu-core returns the error instead of raising it on the device. Reporting through
// the error scope as well would make the caller handle the same failure twice.
#[gpu_test]
static RENDER_PIPELINE_ASYNC_REPORTS_AN_INVALID_DESCRIPTOR_THROUGH_THE_FUTURE:
    GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let vs_module = ctx.device.create_shader_module(TRIVIAL_VERTEX_SHADER_DESC);

        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        // Neither color targets nor a depth-stencil attachment: the pipeline has
        // nothing to render to.
        let pipeline = ctx
            .device
            .create_render_pipeline_async(&wgpu::RenderPipelineDescriptor {
                label: Some("targetless async render pipeline"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &vs_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: None,
                multiview_mask: None,
                cache: None,
            });
        // Pop before awaiting: the guard is neither `Send` nor `Sync`.
        let scope_result = scope.pop();

        let from_future = pipeline.await.err();
        let from_scope = scope_result.await;

        assert!(
            matches!(from_future, Some(wgpu::Error::Validation { .. })),
            "expected a validation error from the future, got {from_future:?}"
        );
        assert!(
            from_scope.is_none(),
            "expected the error scope to stay empty, got {from_scope:?}"
        );
    });

// The compute counterpart of the test above: a vertex entry point can't be used for a
// compute pipeline.
#[gpu_test]
static COMPUTE_PIPELINE_ASYNC_REPORTS_AN_INVALID_DESCRIPTOR_THROUGH_THE_FUTURE:
    GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let vs_module = ctx.device.create_shader_module(TRIVIAL_VERTEX_SHADER_DESC);

        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = ctx
            .device
            .create_compute_pipeline_async(&wgpu::ComputePipelineDescriptor {
                label: Some("vertex entry point async compute pipeline"),
                layout: None,
                module: &vs_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let scope_result = scope.pop();

        let from_future = pipeline.await.err();
        let from_scope = scope_result.await;

        assert!(
            matches!(from_future, Some(wgpu::Error::Validation { .. })),
            "expected a validation error from the future, got {from_future:?}"
        );
        assert!(
            from_scope.is_none(),
            "expected the error scope to stay empty, got {from_scope:?}"
        );
    });

/// Many pipelines begun at once, each with its own program, all resolve, and each draws.
#[gpu_test]
static RENDER_PIPELINES_ASYNC_ALL_RESOLVE_TO_USABLE_PIPELINES: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_async(|ctx| async move {
            let vertex = ctx.device.create_shader_module(TRIVIAL_VERTEX_SHADER_DESC);
            let fragments: Vec<_> = (0..12)
                .map(|colour| coloured_fragment_shader(&ctx, colour))
                .collect();
            let targets = rgba8_target(None);
            let pending: Vec<_> = fragments
                .iter()
                .map(|fragment| {
                    ctx.device
                        .create_render_pipeline_async(&trivial_pipeline_descriptor(
                            &vertex, fragment, None, &targets,
                        ))
                })
                .collect();
            let mut pipelines = Vec::new();
            for pending in pending {
                pipelines.push(
                    resolved(&ctx, pending)
                        .await
                        .expect("an async render pipeline failed"),
                );
            }
            draw_with(&ctx, &pipelines.iter().collect::<Vec<_>>()).await;
        });

/// A pipeline created synchronously that shares its program with one still compiling in the
/// background can be drawn with at once, and the background one still resolves and draws.
#[gpu_test]
static RENDER_PIPELINE_SYNC_SHARING_A_PROGRAM_STILL_LINKING_IS_USABLE_AT_ONCE:
    GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let vertex = ctx.device.create_shader_module(TRIVIAL_VERTEX_SHADER_DESC);
        let fragment = coloured_fragment_shader(&ctx, 7);
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shared layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        let unblended = rgba8_target(None);
        let blended = rgba8_target(Some(wgpu::BlendState::ALPHA_BLENDING));
        let pending = ctx
            .device
            .create_render_pipeline_async(&trivial_pipeline_descriptor(
                &vertex,
                &fragment,
                Some(&layout),
                &unblended,
            ));
        let at_once = ctx
            .device
            .create_render_pipeline(&trivial_pipeline_descriptor(
                &vertex,
                &fragment,
                Some(&layout),
                &blended,
            ));
        draw_with(&ctx, &[&at_once]).await;
        let background = resolved(&ctx, pending)
            .await
            .expect("the background render pipeline failed");
        draw_with(&ctx, &[&background, &at_once]).await;
    });

/// Dropping a pipeline's future before it resolves forgets the pipeline, and the device goes
/// on making and drawing with others.
#[gpu_test]
static RENDER_PIPELINE_ASYNC_DROPPED_BEFORE_IT_RESOLVES_LEAVES_THE_DEVICE_USABLE:
    GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let vertex = ctx.device.create_shader_module(TRIVIAL_VERTEX_SHADER_DESC);
        let dropped = coloured_fragment_shader(&ctx, 3);
        let kept = coloured_fragment_shader(&ctx, 4);
        let targets = rgba8_target(None);
        drop(
            ctx.device
                .create_render_pipeline_async(&trivial_pipeline_descriptor(
                    &vertex, &dropped, None, &targets,
                )),
        );
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        let pipeline = resolved(
            &ctx,
            ctx.device
                .create_render_pipeline_async(&trivial_pipeline_descriptor(
                    &vertex, &kept, None, &targets,
                )),
        )
        .await
        .expect("the kept render pipeline failed");
        draw_with(&ctx, &[&pipeline]).await;
    });

/// A device destroyed while a pipeline compiles resolves it, as an invalid pipeline, without
/// raising anything on the device.
#[gpu_test]
static RENDER_PIPELINE_ASYNC_ON_A_DESTROYED_DEVICE_RESOLVES: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_async(|ctx| async move {
            let vertex = ctx.device.create_shader_module(TRIVIAL_VERTEX_SHADER_DESC);
            let fragment = coloured_fragment_shader(&ctx, 9);
            let targets = rgba8_target(None);
            let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pending = ctx
                .device
                .create_render_pipeline_async(&trivial_pipeline_descriptor(
                    &vertex, &fragment, None, &targets,
                ));
            ctx.device.destroy();
            let from_scope = scope.pop();
            let resolved = resolved(&ctx, pending).await;
            assert!(
                resolved.is_ok(),
                "expected a destroyed device to resolve the pipeline, got {:?}",
                resolved.err()
            );
            assert!(from_scope.await.is_none());
        });

/// On GL, a fragment stage that fails to compile is reported through the future, with the
/// stage that failed, and not through the error scope.
#[gpu_test]
static RENDER_PIPELINE_ASYNC_REPORTS_A_FAILED_COMPILE_THROUGH_THE_FUTURE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(wgpu::Features::PASSTHROUGH_SHADERS)
                .skip(FailureCase::backend(!wgpu::Backends::GL))
                .expect_fail(
                    FailureCase::backend(wgpu::Backends::GL).validation_error("no_such_value"),
                ),
        )
        .run_async(|ctx| async move {
            let (vertex, fragment, layout) = uncompilable_pipeline_parts(&ctx);
            let targets = rgba8_target(None);
            let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Internal);
            let pending = ctx
                .device
                .create_render_pipeline_async(&trivial_pipeline_descriptor(
                    &vertex,
                    &fragment,
                    Some(&layout),
                    &targets,
                ));
            let from_scope = scope.pop();
            let from_future = resolved(&ctx, pending).await.err();
            match from_future {
                Some(wgpu::Error::Internal {
                    ref description, ..
                }) => assert!(
                    description.contains("FRAGMENT"),
                    "expected the failure to name the fragment stage, got {description}"
                ),
                ref other => panic!("expected an internal error from the future, got {other:?}"),
            }
            assert!(from_scope.await.is_none());
        });

/// On GL, the same failure in a pipeline created synchronously is raised on the device, with
/// the stage that failed, as before background compilation.
#[gpu_test]
static RENDER_PIPELINE_SYNC_REPORTS_A_FAILED_COMPILE_THROUGH_THE_ERROR_SCOPE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(wgpu::Features::PASSTHROUGH_SHADERS)
                .skip(FailureCase::backend(!wgpu::Backends::GL))
                .expect_fail(
                    FailureCase::backend(wgpu::Backends::GL).validation_error("no_such_value"),
                ),
        )
        .run_async(|ctx| async move {
            let (vertex, fragment, layout) = uncompilable_pipeline_parts(&ctx);
            let targets = rgba8_target(None);
            let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Internal);
            let _pipeline = ctx
                .device
                .create_render_pipeline(&trivial_pipeline_descriptor(
                    &vertex,
                    &fragment,
                    Some(&layout),
                    &targets,
                ));
            match scope.pop().await {
                Some(wgpu::Error::Internal {
                    ref description, ..
                }) => assert!(
                    description.contains("FRAGMENT"),
                    "expected the failure to name the fragment stage, got {description}"
                ),
                ref other => panic!("expected an internal error on the device, got {other:?}"),
            }
        });
