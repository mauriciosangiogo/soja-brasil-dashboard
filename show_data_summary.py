"""
Script simplificado para mostrar estatísticas dos dados do Supabase
"""

import pandas as pd
from supabase import create_client
from config import SupabaseConfig, AppConfig

print("="*80)
print("🌾 DASHBOARD SOJA BRASIL - RESUMO DOS DADOS")
print("="*80)

print("\n🔄 Conectando ao Supabase...")
try:
    supabase = create_client(SupabaseConfig.get_url(), SupabaseConfig.get_key())
    print("✅ Conexão estabelecida!")
except Exception as e:
    print(f"❌ Erro ao conectar: {e}")
    exit(1)

print("\n📊 Carregando dados...")

# Carregar dados
df_conab = pd.DataFrame(
    supabase.table('df_conab')
    .select("*")
    .gte('ano', AppConfig.ANO_INICIO)
    .lte('ano', AppConfig.ANO_FIM)
    .execute().data
)

df_custos = pd.DataFrame(
    supabase.table('df_custos')
    .select("uf, ano, custo_total, custo_fixo, custo_variavel, total_renda_fatores")
    .gte('ano', AppConfig.ANO_INICIO)
    .lte('ano', AppConfig.ANO_FIM)
    .execute().data
)

df_preco = pd.DataFrame(
    supabase.table('df_preco')
    .select("*")
    .gte('ano', AppConfig.ANO_INICIO)
    .lte('ano', AppConfig.ANO_FIM)
    .execute().data
)

# Merge
df = df_conab.merge(df_custos, on=['uf', 'ano'], how='left')
df = df.merge(df_preco[['ano', 'preco_medio_anual']], on='ano', how='left')

# Criar variáveis derivadas
df['produtividade_sc_ha'] = df['produtividade_kg_ha'] / 60
df['faturamento_ha'] = df['preco_medio_anual'] * df['produtividade_sc_ha']
df['lucro_bruto_ha'] = df['faturamento_ha'] - df['custo_total']
df['roi_percent'] = (df['lucro_bruto_ha'] / df['custo_total']) * 100

print(f"✅ {len(df)} registros carregados!")

print("\n" + "="*80)
print("📈 INFORMAÇÕES GERAIS")
print("="*80)
print(f"📅 Período: {df['ano'].min()} - {df['ano'].max()}")
print(f"🗺️  Estados analisados: {df['uf'].nunique()}")
print(f"📝 Total de observações: {len(df):,}")

print("\n" + "="*80)
print("📊 ESTATÍSTICAS PRINCIPAIS")
print("="*80)

print(f"\n🌾 PRODUTIVIDADE (sc/ha):")
print(f"   • Média:   {df['produtividade_sc_ha'].mean():>8.2f} sc/ha")
print(f"   • Mediana: {df['produtividade_sc_ha'].median():>8.2f} sc/ha")
print(f"   • Mínima:  {df['produtividade_sc_ha'].min():>8.2f} sc/ha")
print(f"   • Máxima:  {df['produtividade_sc_ha'].max():>8.2f} sc/ha")
print(f"   • Desvio:  {df['produtividade_sc_ha'].std():>8.2f} sc/ha")

print(f"\n💰 PREÇO (R$/sc):")
print(f"   • Média:   R$ {df['preco_medio_anual'].mean():>9.2f}/sc")
print(f"   • Mediana: R$ {df['preco_medio_anual'].median():>9.2f}/sc")
print(f"   • Mínimo:  R$ {df['preco_medio_anual'].min():>9.2f}/sc")
print(f"   • Máximo:  R$ {df['preco_medio_anual'].max():>9.2f}/sc")
print(f"   • Desvio:  R$ {df['preco_medio_anual'].std():>9.2f}/sc")

print(f"\n💵 LUCRO BRUTO (R$/ha):")
print(f"   • Média:   R$ {df['lucro_bruto_ha'].mean():>11,.2f}/ha")
print(f"   • Mediana: R$ {df['lucro_bruto_ha'].median():>11,.2f}/ha")
print(f"   • Mínimo:  R$ {df['lucro_bruto_ha'].min():>11,.2f}/ha")
print(f"   • Máximo:  R$ {df['lucro_bruto_ha'].max():>11,.2f}/ha")
print(f"   • Desvio:  R$ {df['lucro_bruto_ha'].std():>11,.2f}/ha")

print(f"\n📈 ROI (%):")
print(f"   • Média:   {df['roi_percent'].mean():>8.2f}%")
print(f"   • Mediana: {df['roi_percent'].median():>8.2f}%")
print(f"   • Mínimo:  {df['roi_percent'].min():>8.2f}%")
print(f"   • Máximo:  {df['roi_percent'].max():>8.2f}%")
print(f"   • Desvio:  {df['roi_percent'].std():>8.2f}%")

print("\n" + "="*80)
print("🏆 TOP 10 ESTADOS POR ROI MÉDIO")
print("="*80)

ranking_roi = df.groupby('uf').agg({
    'roi_percent': 'mean',
    'produtividade_sc_ha': 'mean',
    'lucro_bruto_ha': 'mean'
}).sort_values('roi_percent', ascending=False)

print(f"\n{'#':<4} {'Estado':<6} {'ROI Médio':<12} {'Produt. (sc/ha)':<18} {'Lucro (R$/ha)':<15}")
print("-" * 80)

for i, (estado, row) in enumerate(ranking_roi.head(10).iterrows(), 1):
    roi = row['roi_percent']
    prod = row['produtividade_sc_ha']
    lucro = row['lucro_bruto_ha']

    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    status = "✅" if roi > AppConfig.META_ROI else "⚠️" if roi > 15 else "❌"

    print(f"{emoji} {i:<2} {estado:<6} {status} {roi:>7.2f}%      {prod:>8.2f} sc/ha       R$ {lucro:>11,.2f}")

print("\n" + "="*80)
print("📉 BOTTOM 5 ESTADOS POR ROI MÉDIO")
print("="*80)

print(f"\n{'#':<4} {'Estado':<6} {'ROI Médio':<12} {'Produt. (sc/ha)':<18} {'Lucro (R$/ha)':<15}")
print("-" * 80)

for i, (estado, row) in enumerate(ranking_roi.tail(5).iterrows(), 1):
    roi = row['roi_percent']
    prod = row['produtividade_sc_ha']
    lucro = row['lucro_bruto_ha']

    status = "✅" if roi > AppConfig.META_ROI else "⚠️" if roi > 15 else "❌"

    print(f"   {i:<2} {estado:<6} {status} {roi:>7.2f}%      {prod:>8.2f} sc/ha       R$ {lucro:>11,.2f}")

print("\n" + "="*80)
print("📊 EVOLUÇÃO TEMPORAL (Últimos 5 anos)")
print("="*80)

evolucao = df.groupby('ano').agg({
    'produtividade_sc_ha': 'mean',
    'preco_medio_anual': 'mean',
    'lucro_bruto_ha': 'mean',
    'roi_percent': 'mean'
}).tail(5)

print(f"\n{'Ano':<6} {'Produt.':<12} {'Preço':<12} {'Lucro':<18} {'ROI':<10}")
print("-" * 80)

for ano, row in evolucao.iterrows():
    prod = row['produtividade_sc_ha']
    preco = row['preco_medio_anual']
    lucro = row['lucro_bruto_ha']
    roi = row['roi_percent']

    lucro_status = "📈" if lucro > 0 else "📉"
    roi_status = "✅" if roi > AppConfig.META_ROI else "⚠️" if roi > 15 else "❌"

    print(f"{ano:<6} {prod:>7.2f} sc/ha  R$ {preco:>7.2f}/sc  {lucro_status} R$ {lucro:>9,.2f}  {roi_status} {roi:>6.2f}%")

print("\n" + "="*80)
print("💡 INSIGHTS")
print("="*80)

# Calcular insights
anos_positivos = (df.groupby('ano')['lucro_bruto_ha'].mean() > 0).sum()
total_anos = df['ano'].nunique()
pct_positivos = (anos_positivos / total_anos) * 100

estados_meta = (ranking_roi['roi_percent'] > AppConfig.META_ROI).sum()
total_estados = ranking_roi.shape[0]
pct_meta = (estados_meta / total_estados) * 100

print(f"\n✅ Anos com lucro positivo: {anos_positivos}/{total_anos} ({pct_positivos:.1f}%)")
print(f"🎯 Estados acima da meta de {AppConfig.META_ROI}% ROI: {estados_meta}/{total_estados} ({pct_meta:.1f}%)")
print(f"📊 Variação do preço: {df['preco_medio_anual'].min():.2f} - {df['preco_medio_anual'].max():.2f} R$/sc")
print(f"📈 Variação da produtividade: {df['produtividade_sc_ha'].min():.2f} - {df['produtividade_sc_ha'].max():.2f} sc/ha")

# Correlações principais
corr_preco_lucro = df['preco_medio_anual'].corr(df['lucro_bruto_ha'])
corr_prod_lucro = df['produtividade_sc_ha'].corr(df['lucro_bruto_ha'])

print(f"\n🔗 Correlações com Lucro:")
print(f"   • Preço x Lucro:          {corr_preco_lucro:>6.3f}")
print(f"   • Produtividade x Lucro:  {corr_prod_lucro:>6.3f}")

print("\n" + "="*80)
print("✅ Análise concluída!")
print("="*80)
print("\n💡 Para visualizar os gráficos interativos, execute:")
print("   streamlit run app.py")
print("="*80)
