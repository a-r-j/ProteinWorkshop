# Auxiliary Tasks

Auxiliary tasks define training objectives additional to the main training task.
For instance, these can define auxiliary denoising objectives over sequence, coordinate or angle space.

Auxiliary tasks are implemented by modifying the experiment config to inject the
additional metrics, decoders and losses.
