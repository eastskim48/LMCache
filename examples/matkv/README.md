# MatKV HotpotQA experiment

The default experiment uses `data/LongBench-HotpotQA`, 256-token full chunks,
top-5 retrieval, sequential inference, and disk-backed LMCache storage under
`/mnt/nvme0/dongseob/cache/matkv_hotpotqa`.

`DOCUMENT_BOS` selects the document representation:

- `none`: 256 document tokens.
- `each`: BOS followed by 255 document tokens.

Artifacts and LMCache data are isolated by representation. Partial document
tails are excluded.

## Run

Prepare the Chroma index and the union of chunks retrieved by all queries:

```bash
DOCUMENT_BOS=none examples/matkv/run_prepare.sh
```

Start LMCache in a separate terminal and leave it running:

```bash
DOCUMENT_BOS=none examples/matkv/run_server.sh
```

Materialize only the selected top-5 union:

```bash
DOCUMENT_BOS=none examples/matkv/run_materialize.sh
```

Run sequential MatKV evaluation:

```bash
DOCUMENT_BOS=none examples/matkv/run_eval.sh
```

Repeat all four commands with `DOCUMENT_BOS=each` for the per-document BOS
condition. Vanilla evaluation uses the same retrieved chunks but performs full
prefill without a KV connector. The server and materialization are not needed:

```bash
DOCUMENT_BOS=none examples/matkv/run_vanilla.sh
```

`MODE=baseline examples/matkv/run_eval.sh` remains an alias for compatibility.

Useful smoke-test limits can be forwarded directly:

```bash
examples/matkv/run_materialize.sh --max-chunks 5
examples/matkv/run_eval.sh --max-queries 5
```

The materializer and MatKV evaluator must use the same continuously running
LMCache server. The coordinator is not required unless the server is restarted
between the two phases.
