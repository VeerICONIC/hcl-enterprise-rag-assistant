import io
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import (
    ClientError, EndpointConnectionError, NoCredentialsError, NoRegionError, ProfileNotFound,
)

from app.core import bedrock
from app.core.bedrock import (
    BedrockError, BedrockRuntime, CredentialsError, ModelAccessError, ModelNotFoundError,
    PermissionDeniedError, RegionError, ThrottlingError, create_bedrock_runtime_client,
    create_session, translate_error,
)
from app.core.config import load_settings


def client_error(code, message="msg"):
    return ClientError({"Error": {"Code": code, "Message": message}}, "InvokeModel")


def fake_session(creds=object()):
    sess = MagicMock()
    sess.get_credentials.return_value = creds
    return sess


def test_session_uses_profile_and_region(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "hack")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    session_cls = MagicMock()
    monkeypatch.setattr(bedrock.boto3, "Session", session_cls)
    create_session(load_settings())
    session_cls.assert_called_once_with(profile_name="hack", region_name="eu-west-1")


def test_session_without_profile_uses_default_chain(monkeypatch):
    session_cls = MagicMock()
    monkeypatch.setattr(bedrock.boto3, "Session", session_cls)
    create_session(load_settings())
    session_cls.assert_called_once_with(profile_name=None, region_name="us-east-1")


def test_unknown_profile_is_credentials_error(monkeypatch):
    monkeypatch.setattr(bedrock.boto3, "Session",
                        MagicMock(side_effect=ProfileNotFound(profile="nope")))
    with pytest.raises(CredentialsError, match="AWS_PROFILE"):
        create_session(load_settings())


def test_client_creation_makes_bedrock_runtime_client(monkeypatch):
    sess = fake_session()
    monkeypatch.setattr(bedrock.boto3, "Session", MagicMock(return_value=sess))
    client = create_bedrock_runtime_client(load_settings())
    args, kwargs = sess.client.call_args
    assert args == ("bedrock-runtime",)
    assert kwargs["config"].region_name == "us-east-1"
    assert client is sess.client.return_value


def test_missing_credentials_error(monkeypatch):
    monkeypatch.setattr(bedrock.boto3, "Session", MagicMock(return_value=fake_session(creds=None)))
    with pytest.raises(CredentialsError, match="No AWS credentials"):
        create_bedrock_runtime_client(load_settings())


def test_no_aws_activity_on_construction(monkeypatch):
    session_cls = MagicMock()
    monkeypatch.setattr(bedrock.boto3, "Session", session_cls)
    BedrockRuntime(load_settings())
    session_cls.assert_not_called()


@pytest.mark.parametrize("exc,cls", [
    (NoCredentialsError(), CredentialsError),
    (NoRegionError(), RegionError),
    (EndpointConnectionError(endpoint_url="https://x"), RegionError),
    (client_error("ExpiredTokenException"), CredentialsError),
    (client_error("UnrecognizedClientException"), CredentialsError),
    (client_error("AccessDeniedException", "User is not authorized to perform bedrock:InvokeModel"),
     PermissionDeniedError),
    (client_error("AccessDeniedException", "You don't have access to the model with the specified model ID."),
     ModelAccessError),
    (client_error("ResourceNotFoundException"), ModelNotFoundError),
    (client_error("ValidationException", "The provided model identifier is invalid."), ModelNotFoundError),
    (client_error("ValidationException",
                  "Invocation of model ID with on-demand throughput isn't supported. Use an inference profile."),
     ModelNotFoundError),
    (client_error("ThrottlingException"), ThrottlingError),
    (client_error("InternalServerException", "boom"), BedrockError),
])
def test_error_translation(exc, cls):
    err = translate_error(exc, model_id="m", region="r")
    assert type(err) is cls


def test_runtime_invoke_model_returns_body():
    client = MagicMock()
    client.invoke_model.return_value = {"body": io.BytesIO(b'{"embedding": [1]}')}
    rt = BedrockRuntime(load_settings(), client=client)
    assert rt.invoke_model("m", "{}") == b'{"embedding": [1]}'
    assert client.invoke_model.call_args.kwargs["modelId"] == "m"


def test_runtime_translates_client_errors():
    client = MagicMock()
    client.invoke_model.side_effect = client_error("ResourceNotFoundException")
    client.converse.side_effect = client_error("ThrottlingException")
    rt = BedrockRuntime(load_settings(), client=client)
    with pytest.raises(ModelNotFoundError):
        rt.invoke_model("m", "{}")
    with pytest.raises(ThrottlingError):
        rt.converse("m", messages=[])


def test_runtime_converse_passthrough():
    client = MagicMock()
    client.converse.return_value = {"output": {}}
    rt = BedrockRuntime(load_settings(), client=client)
    assert rt.converse("m", messages=[1]) == {"output": {}}
    client.converse.assert_called_once_with(modelId="m", messages=[1])



def test_account_verification_hold_is_explained_not_called_iam():
    err = translate_error(client_error("AccessDeniedException",
                                       "Your account is currently being verified. Verification normally takes less than 2 hours."),
                          model_id="m", region="us-east-1")
    assert type(err) is PermissionDeniedError
    assert "still verifying this account" in str(err) and "not an IAM" in str(err)
