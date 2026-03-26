# sagemaker-manager

Manage SageMaker resources for Pipecat/Deepgram testing without having to deal with the AWS console.

```bash
cp .env.example .env
# Fill out your .env stuff
uv run sagemaker-manager
uv run sagemaker-manager up # to start stt and tts
uv run sagemaker-manager down --service tts # to stop just tts
```
