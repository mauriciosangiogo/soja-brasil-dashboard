"""
Script: Upload das Tabelas do Projeto Soja para Supabase
Autor: Maurício
Data: 2024

Tabelas:
- df_conab.csv: Produção por estado
- df_custos.csv: Custos de produção por estado
- df_preco.csv: Preços nacionais
"""

import pandas as pd
from sqlalchemy import create_engine, text
import os

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# Connection string do Supabase (substitua [SUA_SENHA])
DATABASE_URL = (
    "postgresql://postgres.pdqoaihshyrnmigymfnd:[soja12245]@"
    "db.pdqoaihshyrnmigymfnd.supabase.co:5432/postgres"
)

# Caminhos dos arquivos (ajuste se necessário)
ARQUIVO_CONAB = "C:\\Users\\ms_sa\\Documents\\projeto-soja-brasil\\data\\processed\\df_conab.csv"
ARQUIVO_CUSTOS = "C:\\Users\\ms_sa\\Documents\\projeto-soja-brasil\\data\\processed\\df_custos.csv"
ARQUIVO_PRECO = "C:\\Users\\ms_sa\\Documents\\projeto-soja-brasil\\data\\processed\\df_preco.csv"

# ============================================================================
# FUNÇÕES
# ============================================================================

def criar_engine():
    """Cria conexão com PostgreSQL"""
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Conexão estabelecida com Supabase!")
        return engine
    except Exception as e:
        print(f"❌ Erro de conexão: {e}")
        raise


def carregar_e_limpar_conab(caminho):
    """
    Carrega dados CONAB e limpa.
    
    Por quê esta função?
        Dados vêm com vírgula como decimal (padrão BR)
        Pandas precisa de ponto como decimal
    """
    print("\n📦 Carregando df_conab.csv...")
    
    df = pd.read_csv(
        caminho,
        sep=';',
        decimal=',',  # Importante! Vírgula como decimal
        encoding='utf-8'
    )
    
    # Renomear para padrão consistente
    df.columns = ['uf', 'safra', 'ano', 'area_mil_ha', 'producao_mil_ton', 
                  'produtividade_kg_ha']
    
    print(f"   ✓ {len(df)} registros carregados")
    print(f"   ✓ Colunas: {list(df.columns)}")
    print(f"   ✓ Estados: {df['uf'].nunique()} ({', '.join(sorted(df['uf'].unique()))})")
    print(f"   ✓ Período: {df['ano'].min()}-{df['ano'].max()}")
    
    return df


def carregar_e_limpar_custos(caminho):
    """Carrega dados de custos e limpa"""
    print("\n📦 Carregando df_custos.csv...")
    
    df = pd.read_csv(
        caminho,
        sep=';',
        decimal=',',
        encoding='utf-8'
    )
    
    # Padronizar nomes de colunas (remover espaços, _rs_ha)
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(' ', '_')
                  .str.replace('_rs_ha', '')
                  .str.replace('_c_rs_ha', '')
                  .str.replace('_b_rs_ha', '')
                  .str.replace('_f_rs_ha', '')
                  .str.replace('_e_rs_ha', ''))
    
    print(f"   ✓ {len(df)} registros carregados")
    print(f"   ✓ Colunas: {len(df.columns)} variáveis de custo")
    print(f"   ✓ Estados: {df['uf'].nunique()}")
    
    return df


def carregar_e_limpar_preco(caminho):
    """Carrega dados de preços"""
    print("\n📦 Carregando df_preco.csv...")
    
    df = pd.read_csv(
        caminho,
        sep=';',
        decimal=',',
        encoding='utf-8'
    )
    
    print(f"   ✓ {len(df)} registros carregados")
    print(f"   ✓ Período: {df['ano'].min()}-{df['ano'].max()}")
    print(f"   ✓ Preço médio: R$ {df['preco_reais'].mean():.2f}/saca")
    
    return df


def criar_tabelas(engine):
    """Cria as três tabelas no banco"""
    print("\n🔨 Criando tabelas no banco...")
    
    sql_tabelas = """
    -- Tabela 1: Produção CONAB
    CREATE TABLE IF NOT EXISTS conab_producao (
        id SERIAL PRIMARY KEY,
        uf VARCHAR(2) NOT NULL,
        safra VARCHAR(10) NOT NULL,
        ano INTEGER NOT NULL,
        area_mil_ha NUMERIC(10, 2),
        producao_mil_ton NUMERIC(10, 2),
        produtividade_kg_ha NUMERIC(10, 2),
        created_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT unique_conab_uf_ano UNIQUE(uf, ano)
    );
    
    -- Tabela 2: Custos de Produção
    CREATE TABLE IF NOT EXISTS custos_producao (
        id SERIAL PRIMARY KEY,
        uf VARCHAR(2) NOT NULL,
        ano INTEGER NOT NULL,
        administrador NUMERIC(10, 2),
        agrotoxicos NUMERIC(10, 2),
        armazenagem NUMERIC(10, 2),
        assistencia_tecnica NUMERIC(10, 2),
        custo_fixo NUMERIC(10, 2),
        custo_operacional NUMERIC(10, 2),
        custo_total NUMERIC(10, 2),
        custo_variavel NUMERIC(10, 2),
        despesas_administrativas NUMERIC(10, 2),
        fertilizantes NUMERIC(10, 2),
        seguro_producao NUMERIC(10, 2),
        sementes NUMERIC(10, 2),
        total_depreciacoes NUMERIC(10, 2),
        total_despesas_custeio NUMERIC(10, 2),
        total_despesas_financeiras NUMERIC(10, 2),
        total_outras_despesas NUMERIC(10, 2),
        total_outros_custos_fixos NUMERIC(10, 2),
        total_renda_fatores NUMERIC(10, 2),
        transporte_externo NUMERIC(10, 2),
        tratores_colheitadeiras NUMERIC(10, 2),
        produtividade NUMERIC(10, 2),
        created_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT unique_custos_uf_ano UNIQUE(uf, ano)
    );
    
    -- Tabela 3: Preços
    CREATE TABLE IF NOT EXISTS precos_soja (
        id SERIAL PRIMARY KEY,
        ano INTEGER NOT NULL UNIQUE,
        preco_reais NUMERIC(10, 2),
        preco_dolar NUMERIC(10, 2),
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- Índices para otimizar consultas
    CREATE INDEX IF NOT EXISTS idx_conab_uf ON conab_producao(uf);
    CREATE INDEX IF NOT EXISTS idx_conab_ano ON conab_producao(ano);
    CREATE INDEX IF NOT EXISTS idx_custos_uf ON custos_producao(uf);
    CREATE INDEX IF NOT EXISTS idx_custos_ano ON custos_producao(ano);
    CREATE INDEX IF NOT EXISTS idx_precos_ano ON precos_soja(ano);
    """
    
    with engine.connect() as conn:
        conn.execute(text(sql_tabelas))
        conn.commit()
    
    print("   ✅ Tabelas criadas:")
    print("      • conab_producao")
    print("      • custos_producao")
    print("      • precos_soja")


def fazer_upload(engine, df, nome_tabela):
    """
    Upload genérico usando Pandas to_sql()
    
    Args:
        engine: Conexão SQLAlchemy
        df: DataFrame a ser enviado
        nome_tabela: Nome da tabela no banco
    """
    print(f"\n📤 Enviando dados para '{nome_tabela}'...")
    
    df.to_sql(
        name=nome_tabela,
        con=engine,
        if_exists='replace',  # Substitui se já existir
        index=False,
        chunksize=100,
        method='multi'
    )
    
    print(f"   ✅ {len(df)} registros enviados com sucesso!")


def verificar_dados(engine):
    """Verifica os dados carregados"""
    print("\n🔍 Verificando dados no banco...")
    
    # Contar registros por tabela
    tabelas = ['conab_producao', 'custos_producao', 'precos_soja']
    
    for tabela in tabelas:
        query = f"SELECT COUNT(*) as total FROM {tabela}"
        result = pd.read_sql(query, engine)
        print(f"   • {tabela}: {result['total'][0]} registros")
    
    # Amostra de dados CONAB
    print("\n📋 Amostra - Produção CONAB (últimos 5 anos):")
    query = """
    SELECT uf, ano, area_mil_ha, producao_mil_ton, produtividade_kg_ha
    FROM conab_producao
    ORDER BY ano DESC, uf
    LIMIT 10
    """
    amostra = pd.read_sql(query, engine)
    print(amostra.to_string(index=False))
    
    # Estatísticas de preços
    print("\n💰 Preços (últimos 5 anos):")
    query = """
    SELECT ano, preco_reais, preco_dolar
    FROM precos_soja
    WHERE ano >= 2019
    ORDER BY ano DESC
    """
    precos = pd.read_sql(query, engine)
    print(precos.to_string(index=False))


def criar_view_consolidada(engine):
    """
    Cria VIEW que junta as 3 tabelas para análise.
    
    Por quê uma VIEW?
        Em vez de fazer JOIN toda vez, criamos uma "tabela virtual"
        que já tem tudo junto. É como fazer um merge() permanente.
    """
    print("\n🔗 Criando VIEW consolidada...")
    
    sql_view = """
    CREATE OR REPLACE VIEW v_dados_consolidados AS
    SELECT 
        c.uf,
        c.ano,
        c.safra,
        c.area_mil_ha,
        c.producao_mil_ton,
        c.produtividade_kg_ha,
        cu.custo_total,
        cu.custo_operacional,
        cu.fertilizantes,
        cu.sementes,
        cu.agrotoxicos,
        p.preco_reais,
        p.preco_dolar,
        -- Cálculos derivados
        (c.area_mil_ha * 1000) as area_ha,
        (c.producao_mil_ton * 1000 * 1000 / 60) as producao_sacas,
        (c.produtividade_kg_ha / 60) as produtividade_sacas_ha,
        ((c.producao_mil_ton * 1000 * 1000 / 60) * p.preco_reais) as receita_bruta,
        (((c.producao_mil_ton * 1000 * 1000 / 60) * p.preco_reais) - 
         (c.area_mil_ha * 1000 * cu.custo_total)) as lucro_bruto
    FROM conab_producao c
    LEFT JOIN custos_producao cu ON c.uf = cu.uf AND c.ano = cu.ano
    LEFT JOIN precos_soja p ON c.ano = p.ano
    WHERE c.ano >= 2008  -- Dados consistentes a partir de 2008
    ORDER BY c.ano DESC, c.uf;
    """
    
    with engine.connect() as conn:
        conn.execute(text(sql_view))
        conn.commit()
    
    print("   ✅ VIEW 'v_dados_consolidados' criada!")
    print("   📊 Use: SELECT * FROM v_dados_consolidados")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Processo completo de upload"""
    print("="*70)
    print("🌾 UPLOAD - PROJETO SOJA BRASIL")
    print("="*70)
    
    # Verificar configuração
    if "[SUA_SENHA]" in DATABASE_URL:
        print("\n⚠️  ERRO: Configure sua senha no DATABASE_URL!")
        print("   Edite a linha 18 do script")
        return
    
    # Verificar se arquivos existem
    for arquivo in [ARQUIVO_CONAB, ARQUIVO_CUSTOS, ARQUIVO_PRECO]:
        if not os.path.exists(arquivo):
            print(f"\n❌ Arquivo não encontrado: {arquivo}")
            print("   Coloque os arquivos CSV na mesma pasta do script")
            return
    
    # 1. Carregar dados
    print("\n" + "="*70)
    print("ETAPA 1: CARREGANDO DADOS")
    print("="*70)
    
    df_conab = carregar_e_limpar_conab(ARQUIVO_CONAB)
    df_custos = carregar_e_limpar_custos(ARQUIVO_CUSTOS)
    df_preco = carregar_e_limpar_preco(ARQUIVO_PRECO)
    
    # 2. Conectar
    print("\n" + "="*70)
    print("ETAPA 2: CONECTANDO AO BANCO")
    print("="*70)
    engine = criar_engine()
    
    # 3. Criar tabelas
    print("\n" + "="*70)
    print("ETAPA 3: CRIANDO ESTRUTURA")
    print("="*70)
    criar_tabelas(engine)
    
    # 4. Upload
    print("\n" + "="*70)
    print("ETAPA 4: ENVIANDO DADOS")
    print("="*70)
    
    fazer_upload(engine, df_conab, 'conab_producao')
    fazer_upload(engine, df_custos, 'custos_producao')
    fazer_upload(engine, df_preco, 'precos_soja')
    
    # 5. Criar VIEW consolidada
    print("\n" + "="*70)
    print("ETAPA 5: CRIANDO VIEW CONSOLIDADA")
    print("="*70)
    criar_view_consolidada(engine)
    
    # 6. Verificar
    print("\n" + "="*70)
    print("ETAPA 6: VERIFICAÇÃO FINAL")
    print("="*70)
    verificar_dados(engine)
    
    # Finalizar
    print("\n" + "="*70)
    print("✨ PROCESSO CONCLUÍDO COM SUCESSO!")
    print("="*70)
    
    print("\n📌 Próximos passos:")
    print("   1. Acesse: https://supabase.com/dashboard")
    print("   2. Vá em: Table Editor")
    print("   3. Você verá suas 3 tabelas + 1 view")
    print("\n💡 Para análises, use:")
    print("   SELECT * FROM v_dados_consolidados")
    print("   (já tem tudo junto e calculado!)")
    
    engine.dispose()


if __name__ == "__main__":
    main()
