from decimal import Decimal
import json
from typing import Optional
from urllib.parse import urljoin
import requests
from core.model_runtime.entities.message_entities import PromptMessage, PromptMessageTool
from core.model_runtime.entities.common_entities import I18nObject
from core.model_runtime.entities.model_entities import AIModelEntity, FetchFrom, ModelPropertyKey, ModelType, PriceConfig
from core.model_runtime.entities.rerank_entities import RerankDocument, RerankResult
from core.model_runtime.errors.invoke import InvokeBadRequestError
from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.model_runtime.model_providers.__base.rerank_model import RerankModel
from core.model_runtime.model_providers.infinity._common import _Infinity_API_Compat


class OAICompatRerankModel(_Infinity_API_Compat, RerankModel):
    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials using requests to ensure compatibility with all providers following OpenAI's API standard.

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            headers = {
                'Content-Type': 'application/json'
            }

            api_key = credentials.get('api_key')
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            endpoint_url = credentials['endpoint_url']
            if not endpoint_url.endswith('/'):
                endpoint_url += '/'
            endpoint_url = urljoin(endpoint_url, 'rerank')

            # prepare the payload for a simple ping to the model
            data = {
                'model': model,
                "query": "Skylark singing to avoid Sparrowhawk attack",
                "documents": [
                    "Beijing is the capital of China",
                    "Larks are passerine birds of the family Alaudidae. Larks have a cosmopolitan distribution with the largest number of species occurring in Africa.",
                    "Hawks are birds of prey of the family Accipitridae. They are very widely distributed and are found on all continents except Antarctica."
                ]
            }

            # send a post request to validate the credentials
            response = requests.post(
                endpoint_url,
                headers=headers,
                json=data,
                timeout=(10, 300)
            )

            if response.status_code != 200:
                raise CredentialsValidateFailedError(
                    f'Credentials validation failed with status code {response.status_code}')

            try:
                json_result = response.json()
            except json.JSONDecodeError as e:
                raise CredentialsValidateFailedError('Credentials validation failed: JSON decode error')

            if ("results" not in json_result) or (len(json_result['results'])!=3):
                raise CredentialsValidateFailedError(
                    'Credentials validation failed: invalid results')

        except CredentialsValidateFailedError:
            raise
        except Exception as ex:
            raise CredentialsValidateFailedError(f'An error occurred during credentials validation: {str(ex)}')
        

    def _invoke(self, model: str, credentials: dict,
                query: str, docs: list[str], score_threshold: Optional[float] = None, top_n: Optional[int] = None,
                user: Optional[str] = None) \
            -> RerankResult:
        """
        Invoke rerank model

        :param model: model name
        :param credentials: model credentials
        :param query: search query
        :param docs: docs for reranking
        :param score_threshold: score threshold
        :param top_n: top n
        :param user: unique user id
        :return: rerank result
        """
        if len(docs) == 0:
            return RerankResult(
                model=model,
                docs=docs
            )

        # Prepare headers and payload for the request
        headers = {
            'Content-Type': 'application/json'
        }

        api_key = credentials.get('api_key')
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        endpoint_url = credentials.get('endpoint_url')
        if not endpoint_url.endswith('/'):
            endpoint_url += '/'

        endpoint_url = urljoin(endpoint_url, 'rerank')


        # prepare the payload
        data = {
            'model': model,
            "query": query,
            "documents": docs,
            "top_n": top_n,
            "return_documents" : False
        }

        # call rerank
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=data,
            timeout=(10, 300)
        )

        if response.status_code != 200:
            raise InvokeBadRequestError(
                f'Failed with status code {response.status_code}')
        json_result = response.json()

        rerank_documents = []
        for idx, result in enumerate(json_result["results"]):
            # format document
            rerank_document = RerankDocument(
                index=result["index"],
                text=data["documents"][result["index"]],
                score=result["relevance_score"],
            )

            # score threshold check
            if score_threshold is not None:
                if result.relevance_score >= score_threshold:
                    rerank_documents.append(rerank_document)
            else:
                rerank_documents.append(rerank_document)

        return RerankResult(
            model=model,
            docs=rerank_documents
        )

    def get_num_tokens(self, model: str, credentials: dict, prompt_messages: list[PromptMessage],
                    tools: Optional[list[PromptMessageTool]] = None) -> int:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return:
        """
        return 0

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        """
            generate custom model entities from credentials
        """

        context_size = credentials.get('context_size')
        try:
            context_size = int(context_size)
        except:
            context_size = 384

        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            model_type=ModelType.RERANK,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: context_size,
            },
            parameter_rules=[],
        )

        return entity









