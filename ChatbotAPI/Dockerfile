# Usar uma imagem base oficial do Python
FROM python:3.10-slim

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia o arquivo de dependências e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação e o frontend HTML para o contêiner
COPY app.py .
COPY index.html .  

# Expor a porta 5000 (onde o Flask roda)
EXPOSE 5000

# Comando para rodar a aplicação quando o contêiner inicia
CMD ["python", "app.py"]