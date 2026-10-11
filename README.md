# simple-softmaxlint

Score source code against a local vLLM server using **Fill-In-The-Middle**, and report the
tokens the model did not expect.

Ported from [softmaxlint](https://github.com/Christian-Andersen/softmaxlint), reduced to a single file under 200 lines of code.

## Example

Scoring a file with an off-by-one boundary bug:

```python
def calculate_average(numbers):
    count = 0
    for i in range(len(numbers)):
        if i <= len(numbers):
            count += numbers[i]
    return count / len(numbers)
```

```console
$ python main.py example.py
4 surprising tokens (of 37 scored):
  line 4    CRITICAL ratio 0.021  rank   2 '<=' -> '<'
```

## Why FIM

Causal log probabilities only ever look **left**. In source code that is the wrong prior:
`if i <= len(numbers)` looks fine given the loop above it, and only betrays itself once the
`total += numbers[i]` below is visible.

FIM fixes the conditioning by reordering the context, so every token is judged with the code
after it as well as before it:

```
<|fim_prefix|>{before}<|fim_suffix|>{after}<|fim_middle|>
```

We read the distribution of the first token the model would emit into the hole. The whole file
goes out as one batch; vLLM schedules the slices concurrently and its prefix cache does most of
the work.

## Running it

Serve a model with the bundled compose file:

```console
$ podman compose up -d              # wait as it takes a while to boot
```

```console
$ uv run python main.py path/to/file.py
```

> Do not substitute a Qwen3 checkpoint. The Qwen3 family declares `<|fim_*|>` tokens but was
> never trained for FIM, so every hole scores `'\n'` and carries no signal. It is not a drop-in
> upgrade despite looking newer.
