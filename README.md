# sagemaker-manager

Manage SageMaker resources for Pipecat/Deepgram testing without having to deal with the AWS console.

```bash
cp .env.example .env
# Fill out your .env stuff

uv run sagemaker-manager              # check endpoint status
uv run sagemaker-manager up           # start both stt and tts
uv run sagemaker-manager up --service stt   # start only stt
uv run sagemaker-manager up --service stt --flux  # start stt using Deepgram Flux model
uv run sagemaker-manager down         # stop both stt and tts
uv run sagemaker-manager down --service tts # stop just tts
```
