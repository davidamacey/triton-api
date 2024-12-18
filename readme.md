# Run Triton Inference Server with Ultralytics

Deploy Triton with docker compose and a FastAPI container for querying the model processing.

## Export Model

Using the export script to get the onnx model to load into the model registry

## Deploy Triton with Ultralytics FastAPI

```bash
docker compose up -d
```

## Scripts

Included are various python scripts to test inference on single, multiple, and batch instances for testing.

## Use Triton

Test with python scripts for single, batch, and multiprocessing

## TO DO: Create FastAPI endpoints for jobs.

## Errors:

- Class names are not matching to the trained data with Triton output
- Speed using batch or multi processing is slow
- Does it need input and output parameters in the config.pbtxt file?


## Start Up Note

Must give time to create the engine files on startup.  Speed up is gained by using the warmup function of config.

Even after the warmup the first inference is still slow to process.  Implementing test images on deployment for user speed to be optimized.