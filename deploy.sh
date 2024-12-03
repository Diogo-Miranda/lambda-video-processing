#!/bin/bash

# Diretório da venv
VENV_DIR="prd"
# Nome do arquivo zip final
ZIP_NAME="lambda_deployment.zip"

# Ativa a venv
source $VENV_DIR/bin/activate

# Cria um diretório temporário
TMP_DIR="tmp_package"
mkdir -p $TMP_DIR

# Copia as dependências para o diretório temporário
cp -r $VENV_DIR/lib/python3.13/site-packages/* $TMP_DIR/

# Entra no diretório temporário e zipa as dependências
cd $TMP_DIR
zip -r9 ../$ZIP_NAME .

# Volta para o diretório original
cd ..

# Adiciona o arquivo lambda_function.py ao zip
zip -g $ZIP_NAME lambda_function.py

# Limpa o diretório temporário
rm -rf $TMP_DIR

echo "Pacote Lambda criado com sucesso: $ZIP_NAME"