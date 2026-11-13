# образ харнесса TechArena, запуск (см. README):
#   docker run --read-only --tmpfs /tmp -e HOME=/tmp -e LLM_BASE_URL -e LLM_API_KEY \
#     -v "$TASKS:/data/tasks:rw" --workdir /app --entrypoint python hack:latest \
#     /app/run_agent.py --tasks-dir /data/tasks
FROM python:3.12.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # во время прогона сети нет, кроме прокси модели, поэтому всё нужное кладём в образ
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    ANONYMIZED_TELEMETRY=False \
    # состояние JiuwenSwarm пишем на tmpfs, корневая ФС только для чтения
    JIUWENSWARM_DATA_DIR=/tmp/jiuwenswarm

# утилиты, которыми агент пользуется в папках задач, python3 уже есть в базовом образе
RUN apt-get update \
 && apt-get install -y --no-install-recommends bash coreutils grep sed gawk findutils procps ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

# кодировки tiktoken скачиваем заранее, чтобы подсчёт токенов работал без сети
RUN mkdir -p /opt/tiktoken && python -c "import tiktoken; [tiktoken.get_encoding(e) for e in ('cl100k_base','o200k_base')]"

COPY run_agent.py ./
COPY harness ./harness
# компилируем при сборке, во время прогона корневая ФС только для чтения
RUN python -m compileall -q /app

ENTRYPOINT ["python"]
CMD ["/app/run_agent.py", "--tasks-dir", "/data/tasks"]
