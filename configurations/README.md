# configurations

We use [Hydra](https://hydra.cc/docs/intro/) to manage configurations. Change/Add the yaml files in this folder
to change the default configurations. You can also override the default configurations by 
passing command line arguments.

All configurations are automatically saved in wandb run.

## Throughput / FLOPs profiling

When `experiment.calculate_throughput` is enabled, the `throughput` node controls how
`OneShotFwBwFLOPsDispatch` attaches hooks to the model. You can provide:

- `target_attr` / `core_module_attr` – where to start and stop the top-level timing.
- `warmup_steps` – how many optimizer steps to skip before capturing metrics.
- `module_specs` – a list of dicts describing per-module hooks (`attr`, optional `hook_type`
	of `module` or `sequence`, and `required`).

Reusable presets for CogVideo-style backbones and FlowViT models live under
`configurations/shortcode/throughput/`. Add the relevant preset to a shortcode's
`defaults` list to enable detailed per-module breakdowns without duplicating config.