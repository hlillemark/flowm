import torch
import functools

def memory_profiler(
    input_shape,
    batch_size=1,
    t=1,
    device='cuda'
):
    """
    A decorator to measure baseline GPU memory usage (forward & backward) by running
    the given model on a dummy input of shape (1,1,...). Then it scales the measured
    peak usage by (batch_size * t), assuming linear memory scaling.
    
    Parameters
    ----------
    input_shape : tuple
        The shape of a single sample (excluding batch and time dimensions).
        For example, (3, 224, 224) or (1024, ).
    batch_size : int
        The real intended batch size for which you want a scaled memory estimate.
    t : int
        The real intended temporal or sequence length dimension for which you want
        a scaled memory estimate.
    device : str
        The device on which to run the memory measurement. Defaults to 'cuda'.

    Usage
    -----
    @memory_profiler((3, 224, 224), batch_size=8, t=4, device='cuda')
    def run_training(model):
        # do your real training code
        return ...

    The decorator will run a quick forward+backward pass with (1,1) shape,
    measure peak memory usage, multiply by (batch_size*t) and print the results.
    Then it calls the original function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(model, *args, **kwargs):

            # Move the model to the desired device (if not already)
            model = model.to(device)
            model.eval()

            # Create a dummy input for baseline measurement
            # We'll do a (1,1) shape plus the rest from input_shape
            if len(input_shape) == 3:
                # e.g. (C, H, W)
                dummy_input_infer = torch.randn((1, 1) + input_shape, device=device)
            else:
                # e.g. (in_features,) or other shape
                dummy_input_infer = torch.randn(tuple([1, 1] + input_shape), device=device)

            # Inference Memory Measurement
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                _ = model(dummy_input_infer)
            peak_inference_base = torch.cuda.max_memory_allocated(device) / (1024**2)

            # Training Memory Measurement
            model.train()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            if len(input_shape) == 3:
                dummy_input_train = torch.randn((1, 1) + input_shape, device=device, requires_grad=True)
            else:
                dummy_input_train = torch.randn((1, 1) + input_shape, device=device, requires_grad=True)

            out = model(dummy_input_train)
            loss = out.mean()  # create a dummy scalar loss
            loss.backward()
            peak_training_base = torch.cuda.max_memory_allocated(device) / (1024**2)

            # Scale up by (batch_size * t) under the linear assumption
            peak_inference_scaled = peak_inference_base * batch_size * t
            peak_training_scaled = peak_training_base * batch_size * t

            print("[memory_profiler] Baseline (bs=1, t=1) peak inference usage: "
                  f"{peak_inference_base:.2f} MB, scaled (bs={batch_size}, t={t}): "
                  f"{peak_inference_scaled:.2f} MB")
            print("[memory_profiler] Baseline (bs=1, t=1) peak training usage:  "
                  f"{peak_training_base:.2f} MB, scaled (bs={batch_size}, t={t}): "
                  f"{peak_training_scaled:.2f} MB")

            # Finally, call the original user function:
            return func(model, *args, **kwargs)
        return wrapper
    return decorator


# Example usage:

if __name__ == "__main__":
    class MySimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc = torch.nn.Linear(32*8*8, 10)

        def forward(self, x):
            # x shape: (batch, t, C, H, W) => flatten them for conv
            b, t_, c, h, w = x.shape
            x = x.view(b*t_, c, h, w)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.nn.functional.adaptive_avg_pool2d(x, (8, 8))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    @memory_profiler((3, 64, 64), batch_size=4, t=2, device='cuda')
    def dummy_train_fn(model):
        """
        Example function that could do real training/inference
        on the given model. Here we just do nothing.
        """
        print("[dummy_train_fn] Running user training code...")

    model = MySimpleModel()
    dummy_train_fn(model)