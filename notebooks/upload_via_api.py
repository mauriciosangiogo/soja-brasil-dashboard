"""
Script: Upload usando Supabase REST API (SEM PostgreSQL direto)
Autor: Maurício
Data: 2024

Vantagens desta abordagem:
- Não precisa de porta 5432 ou 6543
- Funciona em qualquer rede (não tem firewall block)
- Usa HTTPS (porta 443) que sempre funciona
"""

import pandas as pd
from supabase import create_client, Client
import os
from typing import Dict, List
import time

# ============================================================================
# CONFIGURAÇÃO - EDITE AQUI!
# ============================================================================

# Encontre em: Settings > API
SUPABASE_URL = "https://pdqoaihshyrnmigymfnd.supabase.co"
SUPABASE_KEY = "sua_anon_key_aqui"  # Cole a anon/public key

# Arquivos
ARQUIVO_CONAB = "df_conab.csv"
ARQUIVO_CUSTOS = "df_custos.csv"
ARQUIVO_PRECO = "df_preco.csv"

# ============================================================================
# FUNÇÕES
# ============================================================================

def conectar_supabase() -> Client:
    """Conecta ao Supabase via REST API"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Conexão com Supabase estabelecida!")
        return supabase
    except Exception as e:
        print(f"❌ Erro ao conectar: {e}")
        raise


def carregar_conab(caminho):
    """Carrega dados CONAB"""
    print("\n📦 Carregando df_conab.csv...")
    
    df = pd.read_csv(caminho, sep=';', decimal=',', encoding='utf-8')
    
    # Renomear colunas
    df.columns = ['uf', 'safra', 'ano', 'area_mil_ha', 'producao_mil_ton', 
                  'produtividade_kg_ha']
    
    print(f"   ✓ {len(df)} registros")
    print(f"   ✓ Estados: {df['uf'].nunique()}")
    print(f"   ✓ Período: {df['ano'].min()}-{df['ano'].max()}")
    
    return df


def carregar_custos(caminho):
    """Carrega dados de custos"""
    print("\n📦 Carregando df_custos.csv...")
    
    df = pd.read_csv(caminho, sep=';', decimal=',', encoding='utf-8')
    
    # Simplificar nomes das colunas
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(' ', '_')
                  .str.replace('_rs_ha', '')
                  .str.replace('_c_rs_ha', '')
                  .str.replace('_b_rs_ha', '')
                  .str.replace('_f_rs_ha', '')
                  .str.replace('_e_rs_ha', ''))
    
    print(f"   ✓ {len(df)} registros")
    print(f"   ✓ {len(df.columns)} colunas")
    
    return df


def carregar_preco(caminho):
    """Carrega dados de preços"""
    print("\n📦 Carregando df_preco.csv...")
    
    df = pd.read_csv(caminho, sep=';', decimal=',', encoding='utf-8')
    
    print(f"   ✓ {len(df)} registros")
    print(f"   ✓ Preço médio: R$ {df['preco_reais'].mean():.2f}")
    
    return df


def preparar_dados(df: pd.DataFrame) -> List[Dict]:
    """
    Converte DataFrame para lista de dicionários
    (formato aceito pela API Supabase)
    """
    # Substituir NaN por None
    df = df.where(pd.notna(df), None)
    
    # Converter para lista de dicts
    records = df.to_dict(orient='records')
    
    return records


def upload_em_lotes(supabase: Client, tabela: str, dados: List[Dict], 
                     tamanho_lote: int = 100):
    """
    Faz upload em lotes via API Supabase
    
    Por quê em lotes?
        API tem limite de payload. Lotes de 100 são seguros.
    """
    total = len(dados)
    num_lotes = (total + tamanho_lote - 1) // tamanho_lote
    
    print(f"\n📤 Enviando para '{tabela}' ({num_lotes} lotes)...")
    
    for i in range(0, total, tamanho_lote):
        lote = dados[i:i + tamanho_lote]
        lote_num = (i // tamanho_lote) + 1
        
        try:
            # upsert = INSERT ou UPDATE se já existir
            response = supabase.table(tabela).upsert(lote).execute()
            
            print(f"   ✓ Lote {lote_num}/{num_lotes} ({len(lote)} registros)")
            
            # Pausa para não sobrecarregar API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ Erro no lote {lote_num}: {e}")
            raise
    
    print(f"   ✅ {total} registros enviados!")


def verificar_dados(supabase: Client):
    """Verifica dados no banco"""
    print("\n🔍 Verificando dados...")
    
    tabelas = ['conab_producao', 'custos_producao', 'precos_soja']
    
    for tabela in tabelas:
        try:
            response = supabase.table(tabela).select("*", count='exact').execute()
            total = response.count
            print(f"   • {tabela}: {total} registros")
        except Exception as e:
            print(f"   ⚠️  {tabela}: Tabela não existe ainda")
    
    # Amostra
    print("\n📋 Amostra - Produção (últimos 5):")
    try:
        response = supabase.table('conab_producao') \
            .select("uf, ano, area_mil_ha, producao_mil_ton") \
            .order('ano', desc=True) \
            .limit(5) \
            .execute()
        
        for record in response.data:
            print(f"   {record['uf']} {record['ano']}: "
                  f"{record['area_mil_ha']}k ha, "
                  f"{record['producao_mil_ton']}k ton")
    except:
        print("   ⚠️  Ainda sem dados")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Processo completo"""
    print("="*70)
    print("🌾 UPLOAD VIA API - PROJETO SOJA BRASIL")
    print("="*70)
    
    # Verificar configuração
    if SUPABASE_KEY == "sua_anon_key_aqui":
        print("\n⚠️  ERRO: Configure sua SUPABASE_KEY!")
        print("   1. Vá em: Settings > API")
        print("   2. Copie a 'anon/public' key")
        print("   3. Cole na linha 22 do script")
        return
    
    # Verificar arquivos
    for arquivo in [ARQUIVO_CONAB, ARQUIVO_CUSTOS, ARQUIVO_PRECO]:
        if not os.path.exists(arquivo):
            print(f"\n❌ Arquivo não encontrado: {arquivo}")
            return
    
    # 1. Carregar dados
    print("\n" + "="*70)
    print("ETAPA 1: CARREGANDO DADOS")
    print("="*70)
    
    df_conab = carregar_conab(ARQUIVO_CONAB)
    df_custos = carregar_custos(ARQUIVO_CUSTOS)
    df_preco = carregar_preco(ARQUIVO_PRECO)
    
    # 2. Conectar
    print("\n" + "="*70)
    print("ETAPA 2: CONECTANDO VIA API")
    print("="*70)
    supabase = conectar_supabase()
    
    # 3. Preparar dados
    print("\n" + "="*70)
    print("ETAPA 3: PREPARANDO DADOS")
    print("="*70)
    
    dados_conab = preparar_dados(df_conab)
    dados_custos = preparar_dados(df_custos)
    dados_preco = preparar_dados(df_preco)
    
    print(f"   ✓ CONAB: {len(dados_conab)} registros prontos")
    print(f"   ✓ Custos: {len(dados_custos)} registros prontos")
    print(f"   ✓ Preços: {len(dados_preco)} registros prontos")
    
    # 4. Upload
    print("\n" + "="*70)
    print("ETAPA 4: ENVIANDO DADOS")
    print("="*70)
    
    print("\n⚠️  IMPORTANTE:")
    print("   Antes de enviar, você precisa criar as tabelas manualmente")
    print("   no Supabase Table Editor com as seguintes colunas:")
    print("\n   Tabela 1: conab_producao")
    print("   - uf (text)")
    print("   - safra (text)")
    print("   - ano (int8)")
    print("   - area_mil_ha (float8)")
    print("   - producao_mil_ton (float8)")
    print("   - produtividade_kg_ha (float8)")
    
    print("\n   Tabela 2: custos_producao")
    print("   - uf (text)")
    print("   - ano (int8)")
    print("   - [todas as outras colunas como float8]")
    
    print("\n   Tabela 3: precos_soja")
    print("   - ano (int8)")
    print("   - preco_reais (float8)")
    print("   - preco_dolar (float8)")
    
    input("\n⏸️  Pressione ENTER após criar as tabelas no Supabase...")
    
    upload_em_lotes(supabase, 'conab_producao', dados_conab)
    upload_em_lotes(supabase, 'custos_producao', dados_custos)
    upload_em_lotes(supabase, 'precos_soja', dados_preco)
    
    # 5. Verificar
    print("\n" + "="*70)
    print("ETAPA 5: VERIFICAÇÃO")
    print("="*70)
    verificar_dados(supabase)
    
    print("\n" + "="*70)
    print("✨ PROCESSO CONCLUÍDO!")
    print("="*70)
    
    print("\n📌 Próximos passos:")
    print("   1. Acesse: https://supabase.com/dashboard")
    print("   2. Vá em: Table Editor")
    print("   3. Veja suas 3 tabelas com os dados!")


if __name__ == "__main__":
    main()
