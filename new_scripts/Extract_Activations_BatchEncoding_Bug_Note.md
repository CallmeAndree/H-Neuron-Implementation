# Extract Activations BatchEncoding Bug Note

## Summary

A bug happened in `h_neuron_scripts/extract_activations.py` during the forward pass for Qwen.

The error was:

```text
TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not BatchEncoding
```

In simple words:

> The model expected token ID numbers, but the code gave it the whole tokenizer output package instead.

---

## Easy Explanation

Think of the model like a machine that only accepts a list of token numbers.

The model wants something like:

```python
tensor([[151644, 872, 198, 3838, 374, 279, 4226]])
```

But the old code sometimes gave it something like:

```python
{
    "input_ids": tensor([[151644, 872, 198, 3838, 374, 279, 4226]]),
    "attention_mask": tensor([[1, 1, 1, 1, 1, 1, 1]])
}
```

This object is called a `BatchEncoding`.

So the model asked for token numbers, but received the whole tokenizer box.

---

## Analogy

- The model wants a document.
- The tokenizer gives a folder containing the document.
- The old code handed the whole folder to the model.
- The model crashed because it cannot read the folder directly.
- The fixed code opens the folder and gives the model the actual document inside.

---

## Where the Bug Happened

The old code was conceptually like this:

```python
input_ids = tokenizer.apply_chat_template(
    msgs,
    return_tensors="pt",
    add_generation_prompt=False,
).to(model.device)

with torch.no_grad():
    model(input_ids)
```

The problem is that `input_ids` was not always a tensor.

Sometimes it was a `BatchEncoding`, like:

```python
{
    "input_ids": tensor(...),
    "attention_mask": tensor(...)
}
```

Then the call:

```python
model(input_ids)
```

was treated as:

```python
model(input_ids=batch_encoding_object)
```

That means Qwen's embedding layer received the wrong object type.

---

## Why the Error Message Appeared

Inside the model, the embedding layer expects token IDs:

```python
self.embed_tokens(input_ids)
```

The expected type is a tensor:

```python
tensor([[151644, 872, 198, ...]])
```

But the actual type was a `BatchEncoding`:

```python
{
    "input_ids": tensor(...),
    "attention_mask": tensor(...)
}
```

So PyTorch raised:

```text
TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not BatchEncoding
```

Meaning:

> I need token IDs as a tensor, not the full tokenizer output object.

---

## Fix

The fixed code checks what `tokenizer.apply_chat_template()` returns.

If it returns a dictionary-like object, pass it with `**encoding`:

```python
encoding = tokenizer.apply_chat_template(
    msgs,
    return_tensors="pt",
    add_generation_prompt=False,
)

if hasattr(encoding, "keys"):
    encoding = {
        k: v.to(model.device)
        for k, v in encoding.items()
        if torch.is_tensor(v)
    }
    input_ids = encoding["input_ids"]
    with torch.no_grad():
        model(**encoding)
else:
    input_ids = encoding.to(model.device)
    with torch.no_grad():
        model(input_ids=input_ids)
```

Now there are two safe cases:

### Case 1: Tokenizer returns a full package

```python
{
    "input_ids": tensor(...),
    "attention_mask": tensor(...)
}
```

The code calls:

```python
model(**encoding)
```

which becomes:

```python
model(
    input_ids=encoding["input_ids"],
    attention_mask=encoding["attention_mask"],
)
```

### Case 2: Tokenizer returns only token IDs

```python
tensor([[151644, 872, 198, ...]])
```

The code calls:

```python
model(input_ids=input_ids)
```

---

## Final Understanding

The bug was not caused by the model weights or the downloaded files.

The bug was caused by passing the wrong object into the model.

Short version:

> Old code gave the model the whole tokenizer result. Fixed code gives the model the actual token ID tensor.
