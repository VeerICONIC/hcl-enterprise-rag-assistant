import io
import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from app.core.bedrock import BedrockRuntime
from app.core.config import Settings
from app.embeddings.base import EmbeddingError
from app.embeddings.titan import TitanEmbedder


def settings(**kw):
    return Settings(bedrock_embed_dim=kw.pop("dim", 256), **kw)


def client_returning(dim=256, fn=None):
    client = MagicMock()

    def invoke(**kwargs):
        body = json.loads(kwargs["body"])
        vec = fn(body) if fn else [0.1] * dim
        return {"body": io.BytesIO(json.dumps({"embedding": vec, "inputTextTokenCount": 3}).encode())}

    client.invoke_model.side_effect = invoke
    return client


def make(client, **kw):
    s = settings(**kw.pop("s", {}))
    return TitanEmbedder(BedrockRuntime(s, client=client), s, sleep=lambda _: None, **kw)


def err(code):
    return ClientError({"Error": {"Code": code, "Message": "m"}}, "InvokeModel")


def test_request_body_uses_configured_model_and_dimension():
    client = client_returning()
    e = make(client)
    vec = e.embed_query("hello")
    assert len(vec) == 256 and e.dimension == 256
    kwargs = client.invoke_model.call_args.kwargs
    assert kwargs["modelId"] == "amazon.titan-embed-text-v2:0"
    assert json.loads(kwargs["body"]) == {"inputText": "hello", "dimensions": 256, "normalize": True}


def test_documents_and_queries_use_same_model_and_dimension():
    client = client_returning()
    e = make(client)
    e.embed_documents(["a"])
    e.embed_query("b")
    models = {c.kwargs["modelId"] for c in client.invoke_model.call_args_list}
    dims = {json.loads(c.kwargs["body"])["dimensions"] for c in client.invoke_model.call_args_list}
    assert len(models) == 1 and len(dims) == 1


def test_batch_preserves_order_with_concurrency():
    client = client_returning(fn=lambda body: [float(body["inputText"])] * 256)
    e = make(client, s={"embedding_batch_size": 3})
    texts = [str(i) for i in range(10)]
    out = e.embed_documents(texts)
    assert [v[0] for v in out] == [float(i) for i in range(10)]


def test_empty_list_makes_no_calls():
    client = client_returning()
    assert make(client).embed_documents([]) == []
    client.invoke_model.assert_not_called()


@pytest.mark.parametrize("bad", ["", "   \n", None])
def test_empty_text_rejected_without_calling_aws(bad):
    client = client_returning()
    with pytest.raises(EmbeddingError, match="empty"):
        make(client).embed_documents(["ok", bad])
    client.invoke_model.assert_not_called()
    with pytest.raises(EmbeddingError):
        make(client).embed_query(bad)


def test_wrong_dimension_rejected():
    with pytest.raises(EmbeddingError, match="dimension 128, expected BEDROCK_EMBED_DIM=256"):
        make(client_returning(dim=128)).embed_query("x")


@pytest.mark.parametrize("vec", [["a"] * 256, [float("nan")] * 256])
def test_non_numeric_values_rejected(vec):
    client = MagicMock()
    client.invoke_model.return_value = {"body": io.BytesIO(json.dumps({"embedding": vec}).replace("NaN", "NaN").encode())}
    with pytest.raises(EmbeddingError):
        make(client).embed_query("x")


def test_malformed_response():
    client = MagicMock()
    client.invoke_model.return_value = {"body": io.BytesIO(b"not json")}
    with pytest.raises(EmbeddingError, match="unexpected response"):
        make(client).embed_query("x")


def test_throttling_is_retried_then_succeeds():
    client = client_returning()
    good = client.invoke_model.side_effect
    client.invoke_model.side_effect = [err("ThrottlingException"), err("ThrottlingException"), good(body=json.dumps({"inputText": "x"}))]
    assert len(make(client).embed_query("x")) == 256
    assert client.invoke_model.call_count == 3


def test_throttling_gives_up_after_retries():
    client = MagicMock()
    client.invoke_model.side_effect = err("ThrottlingException")
    with pytest.raises(EmbeddingError, match="throttled"):
        make(client, max_retries=2).embed_query("x")
    assert client.invoke_model.call_count == 3


@pytest.mark.parametrize("code", ["AccessDeniedException", "ResourceNotFoundException", "ExpiredTokenException"])
def test_other_errors_are_not_retried_and_wrapped(code):
    client = MagicMock()
    client.invoke_model.side_effect = err(code)
    with pytest.raises(EmbeddingError) as ei:
        make(client).embed_query("x")
    assert client.invoke_model.call_count == 1
    assert ei.value.__cause__ is not None


def test_no_aws_call_on_construction():
    client = MagicMock()
    make(client)
    client.invoke_model.assert_not_called()
