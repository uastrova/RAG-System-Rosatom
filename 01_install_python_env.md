# Подготовка python-окружения 

### 1. Установка основных зависимостей

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

### 2. Подготовка окружения и установка необходимых пакетов

```bash
git clone <репозиторий проекта>
cd ./<папка проекта>

python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

