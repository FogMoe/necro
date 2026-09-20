import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from necro.config import Settings
from necro.engine import DecisionEngine, InputError
from necro.schema import EvaluationRequest, EvaluationResponse


def create_app(settings: Settings | None = None, engine: DecisionEngine | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if engine is None:
        from necro.backend import TransformersScorer

        engine = DecisionEngine(TransformersScorer(settings))
    app = FastAPI(title="Necro", version="0.1.0")
    security = HTTPBearer(auto_error=False)

    def authenticate(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    ):
        if credentials is None or not secrets.compare_digest(
            credentials.credentials, settings.api_key
        ):
            raise HTTPException(
                401, "缺少或无效的 API key。", headers={"WWW-Authenticate": "Bearer"}
            )

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.model_id}

    @app.get("/v1/models", dependencies=[Depends(authenticate)])
    def models():
        return {
            "models": [
                {
                    "name": engine.model_id,
                    "description": "Local Noul, Choice and Score probability judgments.",
                    "release_date": "2026-09-20",
                }
            ]
        }

    @app.post(
        "/v1/systemone", response_model=EvaluationResponse, dependencies=[Depends(authenticate)]
    )
    def system_one(request: EvaluationRequest):
        try:
            return engine.evaluate(request)
        except InputError as error:
            raise HTTPException(422, str(error)) from error
        except RuntimeError as error:
            # 不把模型内部报错或输入原文回传客户端。
            if "out of memory" in str(error).lower():
                raise HTTPException(529, "推理显存不足，请缩短输入或减少批量。") from error
            raise

    return app
