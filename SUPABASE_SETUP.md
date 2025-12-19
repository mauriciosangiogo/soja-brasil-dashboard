# 🔐 Guia de Configuração do Supabase

Este guia explica como configurar a conexão com o Supabase para o Dashboard Soja Brasil.

## 📋 Pré-requisitos

- Conta no [Supabase](https://supabase.com/) (gratuito)
- Python 3.11+
- Projeto criado no Supabase

---

## 🚀 Configuração Rápida

### 1️⃣ Obter Credenciais do Supabase

1. Acesse [https://app.supabase.com/](https://app.supabase.com/)
2. Selecione seu projeto (ou crie um novo)
3. Vá em **Settings** → **API**
4. Copie as seguintes informações:
   - **Project URL** (algo como `https://xxx.supabase.co`)
   - **anon/public key** (token público, seguro para uso no frontend)

### 2️⃣ Configurar as Credenciais

Você tem duas opções:

#### **Opção A: Streamlit Secrets (Recomendado para Streamlit)**

1. Crie/edite o arquivo `.streamlit/secrets.toml`:

```toml
[supabase]
url = "https://SEU-PROJETO.supabase.co"
key = "SUA-CHAVE-PUBLICA-AQUI"
```

2. **IMPORTANTE**: O arquivo `.streamlit/secrets.toml` já está no `.gitignore` e **NUNCA** deve ser commitado!

#### **Opção B: Variáveis de Ambiente**

1. Crie um arquivo `.env` na raiz do projeto:

```bash
SUPABASE_URL=https://SEU-PROJETO.supabase.co
SUPABASE_KEY=SUA-CHAVE-PUBLICA-AQUI
```

2. O arquivo `.env` também está no `.gitignore`

### 3️⃣ Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4️⃣ Executar o Dashboard

```bash
streamlit run app.py
```

---

## 🏗️ Estrutura do Banco de Dados

O dashboard espera as seguintes tabelas no Supabase:

### Tabela: `df_conab`
- Dados de produção e produtividade da CONAB
- Campos principais: `uf`, `ano`, `produtividade_kg_ha`

### Tabela: `df_custos`
- Dados de custos de produção
- Campos principais: `uf`, `ano`, `custo_total`, `custo_fixo`, `custo_variavel`, `total_renda_fatores`

### Tabela: `df_preco`
- Dados de preços históricos
- Campos principais: `ano`, `preco_medio_anual`

---

## 🔒 Segurança

### ✅ Boas Práticas Implementadas

1. **Credenciais não versionadas**:
   - `.streamlit/secrets.toml` e `.env` estão no `.gitignore`
   - Nunca exponha suas chaves no código-fonte

2. **Separação de configurações**:
   - Arquivo `config.py` centraliza todas as configurações
   - Facilita manutenção e deploy

3. **Uso de anon key**:
   - A chave pública (anon key) é segura para uso no frontend
   - As RLS (Row Level Security) policies do Supabase protegem os dados

### ⚠️ NUNCA Faça Isso

- ❌ Não commite `.streamlit/secrets.toml`
- ❌ Não commite arquivos `.env`
- ❌ Não exponha a `service_role_key` no frontend
- ❌ Não compartilhe suas credenciais publicamente

---

## 🌐 Deploy (Streamlit Cloud)

### Para fazer deploy no Streamlit Cloud:

1. Faça push do código para o GitHub (sem as credenciais!)
2. Acesse [share.streamlit.io](https://share.streamlit.io/)
3. Conecte seu repositório
4. Configure os **Secrets** no painel do Streamlit Cloud:
   ```toml
   [supabase]
   url = "https://SEU-PROJETO.supabase.co"
   key = "SUA-CHAVE-PUBLICA-AQUI"
   ```

---

## 🐛 Solução de Problemas

### Erro: "SUPABASE_URL não configurada"

**Solução**: Certifique-se de que o arquivo `.streamlit/secrets.toml` existe e está configurado corretamente.

### Erro: "Table 'df_conab' does not exist"

**Solução**: Verifique se as tabelas foram criadas no Supabase com os nomes corretos.

### Erro de autenticação

**Solução**:
1. Verifique se a chave (key) está correta
2. Confirme que as políticas RLS do Supabase permitem leitura pública

---

## 📚 Recursos Adicionais

- [Documentação Oficial do Supabase](https://supabase.com/docs)
- [Python Client - Supabase](https://supabase.com/docs/reference/python/introduction)
- [Streamlit Secrets Management](https://docs.streamlit.io/develop/concepts/connections/secrets-management)

---

## 🆘 Suporte

Para problemas ou dúvidas:
1. Verifique a [documentação do Supabase](https://supabase.com/docs)
2. Revise os logs do dashboard
3. Entre em contato com o desenvolvedor

---

**Última atualização**: Dezembro 2024
**Autor**: Maurício
