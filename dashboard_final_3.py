"""
dashboard_final.py — Big Four HR Intelligence
Investisseur RH | Data Challenge 2026
Run: python.exe -m streamlit run dashboard_final.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Big Four — HR Intelligence", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# ── Constants ─────────────────────────────────────────────────────────────────
COLORS   = {"Deloitte":"#3B82F6","PwC":"#EF4444","EY":"#F59E0B","KPMG":"#10B981"}
BIG_FOUR = ["Deloitte","PwC","EY","KPMG"]
SRC_MAP  = {"deloitte":"Deloitte","pwc":"PwC","ey":"EY","kpmg":"KPMG"}

GRADE_ORDER  = ["Stagiaire","Junior","Intermédiaire","Senior","Direction"]
GRADE_COLORS = {"Stagiaire":"#93C5FD","Junior":"#60A5FA","Intermédiaire":"#3B82F6",
                "Senior":"#1D4ED8","Direction":"#1E3A8A"}

POS_ORDER = ["Stagiaire Audit","Stagiaire (anonyme)","Consultant Junior","Auditeur Junior",
             "Analyst / Transaction","Consultant","Auditeur Senior","Senior Consultant",
             "Manager","Senior Manager"]
POS_COLORS = {"Stagiaire Audit":"#BAE6FD","Stagiaire (anonyme)":"#7DD3FC",
              "Consultant Junior":"#38BDF8","Auditeur Junior":"#0EA5E9",
              "Analyst / Transaction":"#0284C7","Consultant":"#0369A1",
              "Auditeur Senior":"#075985","Senior Consultant":"#0C4A6E",
              "Manager":"#FB923C","Senior Manager":"#EA580C"}

# Keys loaded dynamically at module level after data load (handles Unicode apostrophes)
_TEMP_DF = None  # will hold df column reference
THEME_LABELS  = ["Autonomie","Travail stimulant","Développement","Ambiance","Management",
                 "Outils","Flexibilité","Équilibre vie pro","Charge travail","Rémunération"]

POSITION_GROUPS = {
    "Stagiaire Audit":["Auditeur Financier Stagiaire","Stagiaire Auditeur financier","Audit Intern",
                       "Financial Auditor Intern","Consultant Stagiaire","Stagiaire Consultant",
                       "Audit Junior","stagiaire audit","Alternant auditeur financier","M&A Analyst Intern"],
    "Stagiaire (anonyme)":["Stagiaire anonyme","Stagiaire","Intern","Auditeur stagiaire"],
    "Auditeur Junior":["Auditeur Financier - Junior","Auditeur Junior","Junior Auditor",
                       "Auditeur Junior Confirmé","Audit Associate","Auditeur Financier","Auditeur"],
    "Consultant Junior":["Junior Consultant","Consultant Junior","Associate Consultant","Associate"],
    "Analyst / Transaction":["Transaction Services Analyst","Analyst","Financial Analyst",
                              "Senior Analyst","Analyste Transaction Services","Transaction Services Associate"],
    "Consultant":["Consultant","Consultante","Financial Auditor","Auditor","Audit","Advisory"],
    "Auditeur Senior":["Auditeur Financier Senior","Senior Auditor","Auditeur Senior","Audit Senior",
                       "Auditeur senior expérimenté","Audit Senior Associate","Senior Associate","Supervisor"],
    "Senior Consultant":["Senior Consultant","Consultant Sénior","Consultant Senior"],
    "Manager":["Manager","Assistant Manager","MANAGER AUDIT","Audit Manager"],
    "Senior Manager":["Senior Manager"],
}

L = dict(template="plotly_white", font=dict(family="Inter,sans-serif",size=11,color="#1f2937"),
         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
         margin=dict(t=45,b=30,l=10,r=10), title_font=dict(size=13,color="#111827"))

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("big_four.csv", low_memory=False)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["cabinet"] = df["source"].map(SRC_MAP)

    def statut(s):
        s=(s or "").lower().replace("\xa0"," ")
        if "stagiaire" in s or "intérimaire" in s: return "Stagiaire/Intérimaire"
        if "actuel" in s: return "Employé actuel"
        if "ancien" in s: return "Ancien employé"
        return "Anonyme"
    df["statut"]     = df["employee_type"].apply(statut)
    df["is_current"] = df["statut"] == "Employé actuel"
    df["date_p"]     = pd.to_datetime(df["date"], errors="coerce")
    df["year"]       = df["date_p"].dt.year
    df["rating_cat"] = pd.cut(df["rating"],bins=[0,2,3,4,5],
        labels=["Négatif (1-2)","Neutre (3)","Positif (4)","Très positif (5)"])
    df["segment"]    = pd.cut(df["rating"],bins=[0,2,3,5],
        labels=["Détracteur","Neutre","Promoteur"])

    def anc(s):
        s=(s or "").lower().replace("\xa0"," ")
        if "moins d'un an" in s or "less than" in s: return "<1 an"
        if "plus d'un an" in s or "1 to 3" in s: return "1-3 ans"
        if "plus de 3 ans" in s or "3 to 5" in s: return "3-5 ans"
        if "plus de 5 ans" in s or "more than 5" in s: return ">5 ans"
        return None
    df["anc_cat"] = df["employee_type"].apply(anc)

    def grade(row):
        gh=str(row.get("grade_hierarchical","")).strip()
        g=str(row.get("grade","")).strip()
        pos=str(row.get("position","")).lower()
        if gh=="Direction" or g in ["Director","Partner","Controller"]: return "Direction"
        if gh=="Intermédiaire": return "Intermédiaire"
        if gh=="Senior" or g in ["Supervisor","Manager"]: return "Senior"
        if gh=="Junior" or g in ["Analyst","Assistant","Associate"]: return "Junior"
        if any(x in pos for x in ["stagiaire","intern","stage","trainee","alternant"]): return "Stagiaire"
        return None
    df["grade_clean"] = df.apply(grade, axis=1)
    df["pos_group"]   = df["position"].apply(
        lambda p: next((g for g,lst in POSITION_GROUPS.items() if p in lst), None))

    # Load sous-thème keys dynamically (handles Unicode apostrophes on all OS)
    return df

df = load()

# Extract sous-thème column names dynamically (handles Unicode apostrophes on all OS)
SOUS_PRO_KEYS = [c for c in df.columns if c.startswith("sous_theme_pro:")]
SOUS_CON_KEYS = [c for c in df.columns if c.startswith("sous_theme_con:")]

# ── Helpers ───────────────────────────────────────────────────────────────────
def filt(df, cabs):
    return df[df["cabinet"].isin(cabs)] if cabs else df

def kpi(col, val, delta="", color="#3B82F6"):
    st.markdown(f"""<div style='background:white;border:1px solid #E5E7EB;border-top:3px solid {color};
        border-radius:10px;padding:16px 18px;box-shadow:0 1px 3px rgba(0,0,0,.05)'>
        <div style='font-size:.6rem;font-weight:800;color:#9CA3AF;text-transform:uppercase;letter-spacing:.12em'>{col}</div>
        <div style='font-size:1.8rem;font-weight:800;color:#111827;line-height:1;margin:4px 0'>{val}</div>
        <div style='font-size:.72rem;color:#6B7280'>{delta}</div></div>""", unsafe_allow_html=True)

def sec(t):
    st.markdown(f"<div style='font-size:.6rem;font-weight:800;color:#9CA3AF;text-transform:uppercase;"
                f"letter-spacing:.15em;margin:1.6rem 0 .3rem'>{t}</div>"
                f"<div style='height:1px;background:#E5E7EB;margin-bottom:.8rem'></div>",
                unsafe_allow_html=True)

def alert(text, kind="info"):
    bg={"info":"#EFF6FF","warn":"#FFFBEB","danger":"#FEF2F2"}[kind]
    br={"info":"#3B82F6","warn":"#F59E0B","danger":"#EF4444"}[kind]
    col={"info":"#1E40AF","warn":"#92400E","danger":"#991B1B"}[kind]
    st.markdown(f"<div style='background:{bg};border-left:4px solid {br};border-radius:0 8px 8px 0;"
                f"padding:10px 14px;font-size:.78rem;color:{col};margin:6px 0'>{text}</div>",
                unsafe_allow_html=True)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*{font-family:'Inter',sans-serif!important}
.stApp{background:#F9FAFB}
#MainMenu,footer,header{visibility:hidden}
[data-testid="stSidebar"]{background:#0F172A!important;border-right:1px solid #1E293B}
[data-testid="stSidebar"] *{color:#94A3B8!important}
.stTabs [data-baseweb="tab-list"]{background:#F1F5F9;border-radius:10px;padding:4px;gap:2px;border:1px solid #E5E7EB}
.stTabs [data-baseweb="tab"]{background:transparent;color:#64748B;border-radius:7px;font-size:.73rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em}
.stTabs [aria-selected="true"]{background:white!important;color:#0F172A!important;box-shadow:0 1px 3px rgba(0,0,0,.08)}
div[data-testid="stMetric"]{background:white;border:1px solid #E5E7EB;border-radius:10px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
div[data-testid="stMetric"] label{color:#9CA3AF!important;font-size:.6rem!important;text-transform:uppercase;letter-spacing:.1em;font-weight:700}
div[data-testid="stMetricValue"]{color:#111827!important;font-size:1.5rem!important;font-weight:800!important}
.stSelectbox>div>div,.stMultiSelect>div>div{border-radius:8px!important;border-color:#E5E7EB!important}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-thumb{background:#CBD5E1;border-radius:2px}
h1,h2,h3{color:#0F172A!important}
</style>""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='padding:10px 0 20px'>
        <div style='font-size:.58rem;color:#334155;text-transform:uppercase;letter-spacing:.2em;font-weight:800'>Data Challenge 2026</div>
        <div style='font-size:1.3rem;font-weight:800;color:#F1F5F9;margin-top:4px;line-height:1.2'>Big Four<br>
        <span style='color:#475569;font-weight:400;font-size:.9rem'>HR Intelligence</span></div>
        <div style='font-size:.65rem;color:#334155;margin-top:6px'>Persona : Investisseur RH</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("", [
        "🏠  Vue Globale",
        "🎓  Par Grade",
        "💼  Par Poste",
        "⚠️   Points de Vigilance",
    ], label_visibility="collapsed")

    st.markdown("---")
    sel_cabs = st.multiselect("CABINETS", BIG_FOUR, default=BIG_FOUR)
    if not sel_cabs: sel_cabs = BIG_FOUR

    sel_statut = st.multiselect("STATUT", ["Employé actuel","Ancien employé","Stagiaire/Intérimaire","Anonyme"],
                                 default=["Employé actuel","Ancien employé","Stagiaire/Intérimaire","Anonyme"])
    if not sel_statut: sel_statut = ["Employé actuel","Ancien employé","Stagiaire/Intérimaire","Anonyme"]

    st.markdown("---")
    st.markdown("""<div style='font-size:.63rem;color:#252545;line-height:2.2'>
        <b style='color:#475569'>SOURCE</b> Glassdoor<br>
        <b style='color:#475569'>AVIS</b> 2 135<br>
        <b style='color:#475569'>PÉRIODE</b> 2019–2024<br>
        <b style='color:#475569'>MÉTHODE</b> EDA + NLP
    </div>""", unsafe_allow_html=True)

# ── Filtered data ─────────────────────────────────────────────────────────────
dff = df[df["cabinet"].isin(sel_cabs) & df["statut"].isin(sel_statut)].copy()
dff_pos   = dff[dff["pos_group"].notna()].copy()
dff_known = dff[dff["grade_clean"].notna()].copy()
pos_avail   = [p for p in POS_ORDER   if p in dff_pos["pos_group"].unique()]
grade_avail = [g for g in GRADE_ORDER if g in dff_known["grade_clean"].unique()]

# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Vue Globale":
# ═══════════════════════════════════════════════════════════════════════════════

    st.markdown("""<div style='background:linear-gradient(135deg,#0F172A,#1E3A5F);border-radius:14px;
        padding:24px 28px;margin-bottom:24px'>
        <div style='font-size:1.5rem;font-weight:800;color:white'>Vue Globale — Attractivité Employeur</div>
        <div style='font-size:.82rem;color:#94A3B8;margin-top:4px'>2 135 avis Glassdoor · EDA + NLP · Investisseur RH · Avril 2026</div>
    </div>""", unsafe_allow_html=True)

    # ── KPIs ──
    sec("KPIs PRINCIPAUX")
    c1,c2,c3,c4,c5 = st.columns(5)
    nm = dff["rating"].mean(); pp=(dff["rating"]>=4).mean()*100
    pn=(dff["rating"]<=2).mean()*100; pc=dff["is_current"].mean()*100
    rn=((dff["recommander"]==1).mean()-(dff["recommander"]==-1).mean())*100
    with c1: kpi("Note Moyenne",f"{nm:.2f}/5",f"Médiane : {dff['rating'].median():.1f}","#3B82F6")
    with c2: kpi("% Positifs ≥4",f"{pp:.1f}%","Ambassadeurs","#10B981")
    with c3: kpi("% Négatifs ≤2",f"{pn:.1f}%","Seuil alerte : 12%","#EF4444")
    with c4: kpi("% Actuels",f"{pc:.1f}%","Proxy rétention","#8B5CF6")
    with c5: kpi("Reco. Nette",f"{rn:+.1f}%","% reco − % non reco","#F59E0B")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Onglets analyse globale ──
    t1,t2,t3,t4,t5,t6 = st.tabs(["📊 Distributions","💬 Sentiments","📈 Rétention","🧠 NLP","🗺️ Risques","🔍 Satisfaction"])

    with t1:
        # Statistiques descriptives interactives
        col_a, col_b = st.columns([1,2])
        with col_a:
            sec("STATISTIQUES DESCRIPTIVES")
            var = st.selectbox("Variable",["rating","recommander","approbation du pdg",
                                           "perspective commerciale","employment_duration"],
                               label_visibility="collapsed")
            v = dff[var].dropna()
            st.metric("Moyenne",   f"{v.mean():.2f}")
            st.metric("Médiane",   f"{v.median():.2f}")
            st.metric("Écart-type",f"{v.std():.2f}")
            st.metric("Min / Max", f"{v.min():.1f} / {v.max():.1f}")
            st.metric("N valides", f"{len(v):,}")
            skew = v.skew()
            st.metric("Asymétrie", f"{skew:.2f}", delta="distribution symétrique" if abs(skew)<0.5 else ("→ queue droite" if skew>0 else "→ queue gauche"))
        with col_b:
            sec("DISTRIBUTION PAR CABINET")
            fig = make_subplots(rows=2,cols=2,subplot_titles=sel_cabs[:4],
                                horizontal_spacing=.08,vertical_spacing=.18)
            nc  = {1:"#FCA5A5",2:"#FDBA74",3:"#FDE68A",4:"#86EFAC",5:"#34D399"}
            for i,cab in enumerate(sel_cabs[:4]):
                r,c = divmod(i,2)
                sub  = dff[dff["cabinet"]==cab]
                dist = sub["rating"].value_counts().sort_index()
                tot  = len(sub)
                fig.add_trace(go.Bar(x=dist.index.astype(str),y=dist.values,
                    marker_color=[nc.get(int(k),"#888") for k in dist.index],
                    marker_line_width=0,
                    text=[f"{v/tot*100:.0f}%" for v in dist.values],
                    textposition="outside",textfont=dict(size=9),showlegend=False),row=r+1,col=c+1)
            fig.update_layout(height=320,**L)
            st.plotly_chart(fig,use_container_width=True)

    with t2:
        col_a,col_b = st.columns(2)
        with col_a:
            sec("RÉPARTITION DES SENTIMENTS")
            sent = dff.groupby(["cabinet","rating_cat"],observed=True).size().reset_index(name="n")
            tot  = dff.groupby("cabinet").size().reset_index(name="tot")
            sent = sent.merge(tot,on="cabinet"); sent["pct"]=sent["n"]/sent["tot"]*100
            fig = px.bar(sent,x="cabinet",y="pct",color="rating_cat",barmode="stack",
                labels={"pct":"% avis","cabinet":"","rating_cat":""},
                color_discrete_map={"Négatif (1-2)":"#EF4444","Neutre (3)":"#F59E0B",
                                     "Positif (4)":"#10B981","Très positif (5)":"#059669"},
                text="pct")
            fig.update_traces(texttemplate="%{text:.0f}%",textposition="inside",textfont=dict(color="white",size=9))
            fig.update_layout(height=300,**L,legend=dict(orientation="h",y=-0.2,font=dict(size=9)))
            st.plotly_chart(fig,use_container_width=True)
        with col_b:
            sec("PROFIL PROMOTEURS vs DÉTRACTEURS")
            seg = dff.groupby(["cabinet","segment"],observed=True).size().reset_index(name="n")
            seg_tot = dff.groupby("cabinet").size().reset_index(name="tot")
            seg = seg.merge(seg_tot,on="cabinet"); seg["pct"]=seg["n"]/seg["tot"]*100
            fig = px.bar(seg,x="cabinet",y="pct",color="segment",barmode="stack",
                labels={"pct":"% avis","cabinet":"","segment":""},
                color_discrete_map={"Détracteur":"#EF4444","Neutre":"#F59E0B","Promoteur":"#10B981"},
                text="pct")
            fig.update_traces(texttemplate="%{text:.0f}%",textposition="inside",textfont=dict(color="white",size=9))
            fig.update_layout(height=300,**L,legend=dict(orientation="h",y=-0.2,font=dict(size=9)))
            st.plotly_chart(fig,use_container_width=True)

        sec("RECOMMANDATION DU CABINET")
        rec = dff.groupby("cabinet").apply(lambda x: pd.Series({
            "Recommande":    (x["recommander"]==1).mean()*100,
            "Neutre":        (x["recommander"]==0).mean()*100,
            "Ne recommande": (x["recommander"]==-1).mean()*100,
        })).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rec["cabinet"],y=rec["Recommande"],name="Recommande ✅",
            marker_color="#10B981",marker_line_width=0,
            text=[f"{v:.0f}%" for v in rec["Recommande"]],textposition="inside",textfont=dict(color="white",size=10)))
        fig.add_trace(go.Bar(x=rec["cabinet"],y=-rec["Ne recommande"],name="Ne recommande pas 🔴",
            marker_color="#EF4444",marker_line_width=0,
            text=[f"{v:.0f}%" for v in rec["Ne recommande"]],textposition="inside",textfont=dict(color="white",size=10)))
        fig.add_hline(y=0,line_color="#374151",line_width=1.5)
        fig.update_layout(barmode="overlay",height=280,yaxis=dict(title="% avis",ticksuffix="%"),
            xaxis_title="",legend=dict(orientation="h",y=-0.2,font=dict(size=10)),**L)
        st.plotly_chart(fig,use_container_width=True)

    with t3:
        col_a,col_b = st.columns(2)
        with col_a:
            sec("NOTE : EMPLOYÉS ACTUELS VS ANCIENS")
            avg_st = (dff[dff["statut"].isin(["Employé actuel","Ancien employé"])]
                      .groupby(["cabinet","statut"])["rating"].mean().reset_index())
            fig = px.bar(avg_st,x="cabinet",y="rating",color="statut",barmode="group",text="rating",
                labels={"rating":"Note /5","cabinet":"","statut":""},
                color_discrete_map={"Employé actuel":"#10B981","Ancien employé":"#EF4444"})
            fig.update_traces(texttemplate="%{text:.2f}",textposition="outside",marker_line_width=0)
            fig.update_layout(yaxis=dict(range=[2.5,4.8]),height=300,
                legend=dict(orientation="h",y=-0.2,font=dict(size=10)),**L)
            st.plotly_chart(fig,use_container_width=True)
            alert("⚠️ <b>PwC (+0,40) et EY (+0,34)</b> ont les écarts actuels/anciens les plus élevés → signal de turnover élevé","warn")
        with col_b:
            sec("NOTE SELON L'ANCIENNETÉ")
            anc_order = ["<1 an","1-3 ans","3-5 ans",">5 ans"]
            anc_df = (dff.dropna(subset=["anc_cat"])
                      .groupby(["cabinet","anc_cat"])["rating"].mean().reset_index())
            fig = px.line(anc_df,x="anc_cat",y="rating",color="cabinet",markers=True,
                labels={"rating":"Note /5","anc_cat":"Ancienneté","cabinet":""},
                color_discrete_map=COLORS,
                category_orders={"anc_cat":anc_order})
            fig.update_traces(line_width=2.5,marker_size=9)
            fig.add_hline(y=3.5,line_dash="dot",line_color="#9CA3AF",
                annotation_text="Seuil 3.5",annotation_font_size=9)
            fig.update_layout(height=300,legend=dict(orientation="h",y=-0.2,font=dict(size=10)),**L)
            st.plotly_chart(fig,use_container_width=True)
            alert("📈 EY se détériore le plus avec l'ancienneté · PwC en légère hausse · KPMG relativement stable","info")

    with t4:
        sec("HEATMAP NLP — SCORE NET PAR THÈME RH ET CABINET")
        col_a,col_b = st.columns([3,2])
        with col_a:
            net_mat = []
            for cab in sel_cabs:
                sub = dff[dff["cabinet"]==cab]; n=len(sub)
                row=[]
                for p,c in zip(SOUS_PRO_KEYS,SOUS_CON_KEYS):
                    row.append(round((sub[p].sum()-sub[c].sum())/n*100,1) if p in sub.columns and n>0 else 0)
                net_mat.append(row)
            net_df = pd.DataFrame(net_mat,index=sel_cabs,columns=THEME_LABELS)
            fig = px.imshow(net_df,text_auto=True,color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,aspect="auto",
                labels={"color":"Score /100 avis"})
            fig.update_layout(height=260,**L,
                coloraxis_colorbar=dict(len=0.8,thickness=12,title=dict(text="")))
            st.plotly_chart(fig,use_container_width=True)
            st.caption("🟢 Force · 🔴 Friction · Valeur = score pour 100 avis (pros − cons)")

        with col_b:
            sec("IMPACT SUR LA NOTE GLOBALE")
            imp_rows=[]
            for p,c,label in zip(SOUS_CON_KEYS,SOUS_CON_KEYS,THEME_LABELS):
                if p not in dff.columns: continue
                wt=dff[dff[p]>0]["rating"].mean(); wot=dff[dff[p]==0]["rating"].mean()
                imp_rows.append({"Thème":label,"Impact":round(wt-wot,2),"N":int((dff[p]>0).sum())})
            imp_df = pd.DataFrame(imp_rows).sort_values("Impact")
            fig = go.Figure(go.Bar(x=imp_df["Impact"],y=imp_df["Thème"],orientation="h",
                marker_color=[("#EF4444" if v<-0.3 else "#F59E0B" if v<0 else "#10B981") for v in imp_df["Impact"]],
                marker_line_width=0,
                text=[f"{v:+.2f}" for v in imp_df["Impact"]],textposition="outside"))
            fig.add_vline(x=-0.3,line_dash="dot",line_color="#EF4444",
                annotation_text="Impact fort",annotation_font_color="#EF4444",annotation_font_size=9)
            fig.update_layout(height=260,xaxis=dict(title="Δ Note"),**L)
            st.plotly_chart(fig,use_container_width=True)
            alert("🔴 <b>Management (−0,99)</b> et <b>Rémunération (−0,42)</b> ont l'impact le plus fort sur la note","danger")

    with t5:
        sec("CARTOGRAPHIE DES RISQUES RH INVESTISSEUR")
        risk_data=[]
        for cab in sel_cabs:
            sub=dff[dff["cabinet"]==cab]; n=len(sub)
            if n<5: continue
            nm=sub["rating"].mean(); pp=(sub["rating"]>=4).mean()*100; pn=(sub["rating"]<=2).mean()*100
            nc=sub[sub["statut"]=="Employé actuel"]["rating"].mean()
            na=sub[sub["statut"]=="Ancien employé"]["rating"].mean()
            ecart = nc-na if not pd.isna(nc) and not pd.isna(na) else 0
            rem_p=SOUS_PRO_KEYS[9]; rem_c=SOUS_CON_KEYS[9]
            rem_net=(sub[rem_p].sum()-sub[rem_c].sum())/n*100 if rem_p in sub.columns else 0
            chg_c=SOUS_CON_KEYS[8]
            chg=(sub[chg_c].sum()/n*100) if chg_c in sub.columns else 0
            sc=(nm/5*0.5+pp/100*0.3+(1-pn/100)*0.2)*100
            risk_data.append({"Cabinet":cab,"Note":nm,"% Négatifs":pn,
                "Écart ret.":ecart,"Score Rémun.":rem_net,"% Charge":chg,"Score":sc})
        risk_df=pd.DataFrame(risk_data)
        col_a,col_b=st.columns(2)
        with col_a:
            fig=px.scatter(risk_df,x="Score Rémun.",y="Note",size="% Négatifs",color="Cabinet",text="Cabinet",
                title="Note × Rémunération × % Négatifs (taille bulle)",
                labels={"Score Rémun.":"Score net rémunération","Note":"Note /5"},
                color_discrete_map=COLORS)
            fig.add_vline(x=0,line_dash="dot",line_color="#9CA3AF")
            fig.add_hline(y=3.7,line_dash="dot",line_color="#9CA3AF")
            fig.add_annotation(x=risk_df["Score Rémun."].min()*0.9,y=risk_df["Note"].min()*0.995,
                text="⚠️ Zone risque",showarrow=False,font=dict(color="#EF4444",size=10))
            fig.update_traces(textposition="top center",marker=dict(sizemin=10))
            fig.update_layout(height=300,showlegend=False,**L)
            st.plotly_chart(fig,use_container_width=True)
        with col_b:
            fig=px.bar(risk_df.sort_values("Score"),x="Score",y="Cabinet",orientation="h",
                title="Score composite d'attractivité",
                color="Cabinet",color_discrete_map=COLORS,
                text=[f"{v:.1f}" for v in risk_df.sort_values("Score")["Score"]])
            fig.update_traces(textposition="outside",marker_line_width=0)
            fig.add_vline(x=75,line_dash="dot",line_color="#9CA3AF",
                annotation_text="Seuil 75",annotation_font_size=9)
            fig.update_layout(height=300,xaxis=dict(range=[60,90]),showlegend=False,**L)
            st.plotly_chart(fig,use_container_width=True)
        st.caption("Score composite : Note×50% + %Positifs×30% + (1−%Négatifs)×20%")

    with t6:
        sec("FACTEURS DÉTERMINANTS DE LA SATISFACTION")
        col_a,col_b=st.columns(2)
        with col_a:
            # Spearman correlations
            num_candidates = ["recommander","approbation du pdg","perspective commerciale"]
            num_candidates += [c for c in SOUS_PRO_KEYS if c in dff.columns]
            sp_rows=[]
            for col in num_candidates:
                if col not in dff.columns: continue
                try:
                    r,p=stats.spearmanr(dff["rating"].dropna(),dff[col].dropna(),nan_policy="omit")
                    label=col.replace("sous_theme_pro: ","").replace("/ confiance","")[:30]
                    sp_rows.append({"Variable":label,"r":round(r,3)})
                except: pass
            sp_df=pd.DataFrame(sp_rows).sort_values("r",key=abs,ascending=True).tail(12)
            fig=go.Figure(go.Bar(x=sp_df["r"],y=sp_df["Variable"],orientation="h",
                marker_color=["#10B981" if v>0 else "#EF4444" for v in sp_df["r"]],
                marker_line_width=0,
                text=[f"{v:+.3f}" for v in sp_df["r"]],textposition="outside"))
            fig.add_vline(x=0,line_color="#374151",line_width=1)
            fig.update_layout(title="Corrélations Spearman avec la note",height=340,
                xaxis=dict(title="Corrélation"),**L)
            st.plotly_chart(fig,use_container_width=True)
        with col_b:
            # Score note × approbation PDG
            fig=make_subplots(rows=1,cols=2,subplot_titles=["Approbation PDG vs Note","Perspective com. vs Note"])
            for cab in sel_cabs:
                sub=dff[dff["cabinet"]==cab]
                pdg=sub.groupby("approbation du pdg")["rating"].mean().reset_index()
                com=sub.groupby("perspective commerciale")["rating"].mean().reset_index()
                fig.add_trace(go.Scatter(x=pdg["approbation du pdg"],y=pdg["rating"],
                    mode="markers+lines",name=cab,line=dict(color=COLORS[cab],width=2),
                    marker=dict(size=9,color=COLORS[cab])),row=1,col=1)
                fig.add_trace(go.Scatter(x=com["perspective commerciale"],y=com["rating"],
                    mode="markers+lines",name=cab,line=dict(color=COLORS[cab],width=2),
                    marker=dict(size=9,color=COLORS[cab]),showlegend=False),row=1,col=2)
            fig.update_xaxes(tickvals=[-1,0,1],ticktext=["Nég","Neutre","Pos"])
            fig.update_yaxes(range=[2.5,4.8],title_text="Note /5")
            fig.update_layout(height=340,legend=dict(orientation="h",y=-0.2,font=dict(size=9)),**L)
            st.plotly_chart(fig,use_container_width=True)

        # Score composite investisseur
        sec("SCORE D'ATTRACTIVITÉ INVESTISSEUR — VUE RADAR")
        AMB_P=SOUS_PRO_KEYS[3]; AMB_C=SOUS_CON_KEYS[3]
        dims=["Note /5\n(norm.)","% Avis\npositifs","% Actuels","% Reco.\nnet","Score\nAmbiance"]
        fig=go.Figure()
        for cab in sel_cabs:
            sub=dff[dff["cabinet"]==cab]; n=len(sub)
            if n<5: continue
            r_,g_,b_=int(COLORS[cab][1:3],16),int(COLORS[cab][3:5],16),int(COLORS[cab][5:7],16)
            amb=min(max((sub[AMB_P].sum()-sub[AMB_C].sum())/n*100+50,0),100)
            vals=[sub["rating"].mean()/5*100,(sub["rating"]>=4).mean()*100,
                  sub["is_current"].mean()*100,
                  ((sub["recommander"]==1).mean()-(sub["recommander"]==-1).mean())*100,amb]
            vals+=[vals[0]]
            fig.add_trace(go.Scatterpolar(r=vals,theta=dims+[dims[0]],fill="toself",name=cab,
                line=dict(color=COLORS[cab],width=2.5),fillcolor=f"rgba({r_},{g_},{b_},.10)"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100],gridcolor="#E5E7EB")),
            height=380,**L,legend=dict(orientation="h",y=-0.1,font=dict(size=11)))
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎓  Par Grade":
# ═══════════════════════════════════════════════════════════════════════════════

    st.markdown("""<div style='background:linear-gradient(135deg,#0F172A,#1E3A5F);border-radius:14px;
        padding:24px 28px;margin-bottom:24px'>
        <div style='font-size:1.5rem;font-weight:800;color:white'>Analyse par Grade Hiérarchique</div>
        <div style='font-size:.82rem;color:#94A3B8;margin-top:4px'>Stagiaire · Junior · Intermédiaire · Senior · Direction</div>
    </div>""", unsafe_allow_html=True)

    t1,t2,t3,t4,t5 = st.tabs(["📊 Notes","😊 Sentiment","🧠 Thèmes NLP","💰 Rémunération","💼 Recommandations"])

    with t1:
        col_a,col_b=st.columns(2)
        with col_a:
            sec("NOTE MOYENNE PAR GRADE ET CABINET")
            avg_gc=(dff_known.groupby(["grade_clean","cabinet"])["rating"]
                    .agg(["mean","count"]).reset_index())
            avg_gc.columns=["Grade","Cabinet","Note moy","Nb avis"]
            avg_gc=avg_gc[avg_gc["Nb avis"]>=3]
            pivot=avg_gc.pivot(index="Grade",columns="Cabinet",values="Note moy").reindex(grade_avail)
            fig=px.imshow(pivot,text_auto=".2f",color_continuous_scale="RdYlGn",
                color_continuous_midpoint=3.5,aspect="auto",labels={"color":"Note /5"})
            fig.update_layout(height=310,**L)
            st.plotly_chart(fig,use_container_width=True)
        with col_b:
            sec("BOX PLOT — NOTES PAR GRADE")
            fig=px.box(dff_known,x="grade_clean",y="rating",color="grade_clean",
                labels={"rating":"Note /5","grade_clean":""},
                color_discrete_map=GRADE_COLORS,
                category_orders={"grade_clean":grade_avail},points="outliers")
            fig.update_layout(showlegend=False,height=310,**L)
            st.plotly_chart(fig,use_container_width=True)

        sec("NOTE PAR GRADE × CABINET — BAR CHART")
        avg_gc2=avg_gc[avg_gc["Nb avis"]>=5]
        fig=px.bar(avg_gc2,x="Grade",y="Note moy",color="Cabinet",barmode="group",text="Note moy",
            labels={"Note moy":"Note /5","Grade":""},color_discrete_map=COLORS,
            category_orders={"Grade":grade_avail})
        fig.update_traces(texttemplate="%{text:.2f}",textposition="outside",marker_line_width=0)
        fig.update_layout(yaxis=dict(range=[2.5,5.5]),height=320,
            legend=dict(orientation="h",y=-0.15,font=dict(size=10)),**L)
        st.plotly_chart(fig,use_container_width=True)

    with t2:
        col_a,col_b=st.columns(2)
        with col_a:
            sec("RECOMMANDATION PAR GRADE")
            rec_g=(dff_known.assign(rec=dff_known["recommander"].map({1:"Recommande ✅",0:"Neutre ➖",-1:"Ne recommande pas 🔴"}))
                   .groupby(["grade_clean","rec"]).size().reset_index(name="n"))
            fig=px.bar(rec_g,x="grade_clean",y="n",color="rec",barmode="stack",
                labels={"n":"Nb avis","grade_clean":"","rec":""},
                color_discrete_map={"Recommande ✅":"#10B981","Neutre ➖":"#F59E0B","Ne recommande pas 🔴":"#EF4444"},
                category_orders={"grade_clean":grade_avail})
            fig.update_layout(height=310,legend=dict(orientation="h",y=-0.2,font=dict(size=9)),**L)
            st.plotly_chart(fig,use_container_width=True)
        with col_b:
            sec("SENTIMENT DIVERGENT PAR GRADE")
            sg=dff_known.groupby("grade_clean").apply(lambda x: pd.Series({
                "% Positifs":(x["rating"]>=4).mean()*100,
                "% Négatifs":(x["rating"]<=2).mean()*100})).reindex(grade_avail).dropna().reset_index()
            fig=go.Figure()
            fig.add_trace(go.Bar(x=sg["grade_clean"],y=sg["% Positifs"],name="% Positifs ≥4",
                marker_color="#10B981",marker_line_width=0))
            fig.add_trace(go.Bar(x=sg["grade_clean"],y=-sg["% Négatifs"],name="% Négatifs ≤2",
                marker_color="#EF4444",marker_line_width=0))
            fig.add_hline(y=0,line_color="#374151",line_width=1.5)
            fig.update_layout(barmode="overlay",height=310,
                yaxis=dict(title="% avis",ticksuffix="%"),xaxis_title="",
                legend=dict(orientation="h",y=-0.2,font=dict(size=9)),**L)
            st.plotly_chart(fig,use_container_width=True)

    with t3:
        sec("HEATMAP COMPLÈTE — SCORE NET PAR GRADE × 10 SOUS-THÈMES RH")
        mat_rows=[]
        for grade in grade_avail:
            sub=dff_known[dff_known["grade_clean"]==grade]
            if len(sub)<3: continue
            n=len(sub); row={"Grade":grade}
            for p,c,label in zip(SOUS_PRO_KEYS,SOUS_CON_KEYS,THEME_LABELS):
                if p in sub.columns and c in sub.columns:
                    row[label]=round((sub[p].sum()-sub[c].sum())/n*100,1)
            mat_rows.append(row)
        if mat_rows:
            mat_df=pd.DataFrame(mat_rows).set_index("Grade")
            nc=[c for c in mat_df.columns if c!="Nb avis"]
            fig=px.imshow(mat_df[nc].T,text_auto=".1f",color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,aspect="auto",
                labels={"color":"Score /100","x":"Grade","y":"Thème"})
            fig.update_layout(height=400,**L)
            st.plotly_chart(fig,use_container_width=True)
            st.caption("🟢 Force · 🔴 Friction · Valeur = score pour 100 avis · Lire par colonne = profil complet d'un grade")

    with t4:
        sec("ÉQUILIBRE RÉMUNÉRATION — AVANTAGE VS INCONVÉNIENT PAR GRADE")
        RP=SOUS_PRO_KEYS[9]; RC=SOUS_CON_KEYS[9]
        rem_rows=[]
        for grade in grade_avail:
            sub=dff_known[dff_known["grade_clean"]==grade]
            if len(sub)<3 or RP not in sub.columns: continue
            n=len(sub)
            rem_rows.append({"Grade":grade,"% avantage":sub[RP].sum()/n*100,
                "% inconvénient":sub[RC].sum()/n*100,"Score net":(sub[RP].sum()-sub[RC].sum())/n*100})
        rem_df=pd.DataFrame(rem_rows)
        col_a,col_b=st.columns(2)
        with col_a:
            fig=go.Figure()
            fig.add_trace(go.Bar(x=rem_df["Grade"],y=rem_df["% avantage"],name="En avantage",
                marker_color="#10B981",marker_line_width=0))
            fig.add_trace(go.Bar(x=rem_df["Grade"],y=-rem_df["% inconvénient"],name="En inconvénient",
                marker_color="#EF4444",marker_line_width=0))
            fig.add_hline(y=0,line_color="#374151",line_width=1.5)
            fig.update_layout(barmode="overlay",title="Rémunération : pros ↑ vs cons ↓",height=300,
                yaxis=dict(title="% avis",ticksuffix="%"),legend=dict(orientation="h",y=-0.25,font=dict(size=9)),**L)
            st.plotly_chart(fig,use_container_width=True)
        with col_b:
            fig=go.Figure(go.Bar(x=rem_df["Score net"],y=rem_df["Grade"],orientation="h",
                marker_color=["#10B981" if v>=0 else "#EF4444" for v in rem_df["Score net"]],
                marker_line_width=0,text=[f"{v:+.1f}" for v in rem_df["Score net"]],textposition="outside"))
            fig.add_vline(x=0,line_color="#374151",line_width=1.5)
            fig.update_layout(title="Score net Rémunération par grade",height=300,
                xaxis=dict(title="Score net /100 avis"),**L)
            st.plotly_chart(fig,use_container_width=True)
        alert("🔴 Les <b>Juniors</b> ont le score rémunération le plus négatif (−20,5) — risque de turnover précoce","danger")

    with t5:
        sec("RECOMMANDATIONS PAR GRADE — INVESTISSEUR RH")
        recs={
            "Stagiaire":     ("🟢","Levier recrutement","Note 4,17/5 — satisfaction élevée","Structurer la conversion stagiaire → CDI · Programme de mentoring"),
            "Junior":        ("🔴","Risque élevé","Rémunération −20,5 · Charge 47%","Réviser la grille salariale junior · Benchmarking marché"),
            "Intermédiaire": ("🟡","Vigilance","Friction autonomie et management","Délégation ciblée · Programmes leadership intermédiaire"),
            "Senior":        ("🟡","Vigilance","Charge travail 49,6% des inconvénients","Flexibilité horaire · Politiques de modulation"),
            "Direction":     ("🟢","Favorable","Taux de recommandation 78,9%","Maintenir la communication stratégique · Surveiller KPMG Direction (3,44)"),
        }
        cols=st.columns(len(grade_avail))
        for i,grade in enumerate(grade_avail):
            if grade not in recs: continue
            icon,signal,detail,action=recs[grade]
            border={"🟢":"#10B981","🟡":"#F59E0B","🔴":"#EF4444"}[icon]
            with cols[i]:
                st.markdown(f"""<div style='background:white;border:1px solid #E5E7EB;border-top:3px solid {border};
                    border-radius:10px;padding:16px;min-height:160px'>
                    <div style='font-size:.62rem;font-weight:800;color:#9CA3AF;text-transform:uppercase;letter-spacing:.1em'>{grade}</div>
                    <div style='font-size:.95rem;font-weight:700;color:#111827;margin:4px 0'>{icon} {signal}</div>
                    <div style='font-size:.7rem;color:#6B7280;margin-bottom:6px'>{detail}</div>
                    <div style='font-size:.68rem;color:{border};font-weight:700'>→ {action}</div>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💼  Par Poste":
# ═══════════════════════════════════════════════════════════════════════════════

    st.markdown("""<div style='background:linear-gradient(135deg,#0F172A,#1E3A5F);border-radius:14px;
        padding:24px 28px;margin-bottom:24px'>
        <div style='font-size:1.5rem;font-weight:800;color:white'>Analyse par Type de Poste</div>
        <div style='font-size:.82rem;color:#94A3B8;margin-top:4px'>Stagiaire · Auditeur Junior · Consultant · Senior Consultant · Manager...</div>
    </div>""", unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["📊 Notes & Sentiment","🗺️ Heatmaps","⚠️ Risques","💼 Recommandations"])

    with t1:
        sec("NOTE MOYENNE PAR POSTE")
        avg_p=(dff_pos.groupby("pos_group")["rating"].agg(["mean","count"])
               .reindex(pos_avail).dropna().reset_index())
        avg_p.columns=["Poste","Moyenne","Nb avis"]
        fig=go.Figure()
        for _,row in avg_p.iterrows():
            fig.add_trace(go.Bar(x=[row["Poste"]],y=[row["Moyenne"]],
                marker_color=POS_COLORS.get(row["Poste"],"#64748B"),marker_line_width=0,
                text=f'{row["Moyenne"]:.2f}',textposition="outside",
                name=row["Poste"],showlegend=False))
        fig.add_hline(y=dff_pos["rating"].mean(),line_dash="dot",line_color="#9CA3AF",
            annotation_text=f"Moy. : {dff_pos['rating'].mean():.2f}",annotation_font_size=9)
        fig.update_layout(xaxis=dict(tickangle=-30),yaxis=dict(range=[2.5,5.2],title="Note /5"),
            height=320,**L)
        st.plotly_chart(fig,use_container_width=True)

        col_a,col_b=st.columns(2)
        with col_a:
            sec("BOX PLOT — NOTES PAR POSTE")
            fig=px.box(dff_pos[dff_pos["pos_group"].isin(pos_avail)],
                x="pos_group",y="rating",color="pos_group",
                labels={"rating":"Note /5","pos_group":""},
                color_discrete_map=POS_COLORS,
                category_orders={"pos_group":pos_avail},points="outliers")
            fig.update_layout(showlegend=False,xaxis_tickangle=-30,height=300,**L)
            st.plotly_chart(fig,use_container_width=True)
        with col_b:
            sec("SENTIMENT DIVERGENT PAR POSTE")
            sp=dff_pos[dff_pos["pos_group"].isin(pos_avail)].groupby("pos_group").apply(
                lambda x: pd.Series({"% Positifs":(x["rating"]>=4).mean()*100,
                                      "% Négatifs":(x["rating"]<=2).mean()*100})
            ).reindex(pos_avail).dropna().reset_index()
            fig=go.Figure()
            fig.add_trace(go.Bar(x=sp["pos_group"],y=sp["% Positifs"],name="% Positifs ≥4",
                marker_color="#10B981",marker_line_width=0))
            fig.add_trace(go.Bar(x=sp["pos_group"],y=-sp["% Négatifs"],name="% Négatifs ≤2",
                marker_color="#EF4444",marker_line_width=0))
            fig.add_hline(y=0,line_color="#374151",line_width=1.5)
            fig.update_layout(barmode="overlay",height=300,
                yaxis=dict(ticksuffix="%"),xaxis_tickangle=-30,
                legend=dict(orientation="h",y=-0.25,font=dict(size=9)),**L)
            st.plotly_chart(fig,use_container_width=True)

    with t2:
        sec("HEATMAP : NOTE PAR POSTE × CABINET")
        avg_pc=(dff_pos.groupby(["pos_group","cabinet"])["rating"].agg(["mean","count"]).reset_index())
        avg_pc.columns=["Poste","Cabinet","Note moy","Nb avis"]
        avg_pc=avg_pc[(avg_pc["Nb avis"]>=3)&(avg_pc["Poste"].isin(pos_avail))]
        pivot_pc=avg_pc.pivot(index="Poste",columns="Cabinet",values="Note moy").reindex(pos_avail)
        fig=px.imshow(pivot_pc,text_auto=".2f",color_continuous_scale="RdYlGn",
            color_continuous_midpoint=3.5,aspect="auto",labels={"color":"Note /5"})
        fig.update_layout(height=380,**L)
        st.plotly_chart(fig,use_container_width=True)

        sec("HEATMAP COMPLÈTE — SCORE NET PAR POSTE × 10 SOUS-THÈMES RH ⭐")
        mat_rows=[]
        for pos in pos_avail:
            sub=dff_pos[dff_pos["pos_group"]==pos]
            if len(sub)<5: continue
            n=len(sub); row={"Poste":pos}
            for p,c,label in zip(SOUS_PRO_KEYS,SOUS_CON_KEYS,THEME_LABELS):
                if p in sub.columns and c in sub.columns:
                    row[label]=round((sub[p].sum()-sub[c].sum())/n*100,1)
            mat_rows.append(row)
        if mat_rows:
            mat_df=pd.DataFrame(mat_rows).set_index("Poste")
            nc=[c for c in mat_df.columns if c!="Nb avis"]
            fig=px.imshow(mat_df[nc].T,text_auto=".1f",color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,aspect="auto",
                labels={"color":"Score /100","x":"Poste","y":"Thème"})
            fig.update_layout(xaxis_tickangle=-30,height=420,**L)
            st.plotly_chart(fig,use_container_width=True)
            st.caption("🟢 Force · 🔴 Friction · Valeur = score pour 100 avis · Lire par colonne = profil complet d'un poste")

    with t3:
        sec("CLASSEMENT DES POSTES PAR SCORE COMPOSITE — RISQUE RH")
        def get_net(sub,p,c):
            n=len(sub)
            return round((sub[p].sum()-sub[c].sum())/n*100,1) if n>0 and p in sub.columns else float("nan")
        synth=[]
        for pos in pos_avail:
            sub=dff_pos[dff_pos["pos_group"]==pos]
            if len(sub)<5: continue
            n=len(sub)
            sc=((sub["rating"].mean()/5*0.40+(sub["rating"]>=4).mean()*0.25+
                 (1-(sub["rating"]<=2).mean())*0.15+(sub["recommander"]==1).mean()*0.20)*100)
            synth.append({"Poste":pos,"Nb avis":n,"Note":round(sub["rating"].mean(),2),
                "% Pos":round((sub["rating"]>=4).mean()*100,1),
                "% Nég":round((sub["rating"]<=2).mean()*100,1),
                "Score Rémun.":get_net(sub,SOUS_PRO_KEYS[9],SOUS_CON_KEYS[9]),
                "Score Charge":get_net(sub,SOUS_PRO_KEYS[8],SOUS_CON_KEYS[8]),
                "Score":round(sc,1)})
        synth_df=pd.DataFrame(synth).sort_values("Score",ascending=True)

        col_a,col_b=st.columns([2,3])
        with col_a:
            fig=go.Figure(go.Bar(x=synth_df["Score"],y=synth_df["Poste"],orientation="h",
                marker_color=[POS_COLORS.get(p,"#64748B") for p in synth_df["Poste"]],
                marker_line_width=0,
                text=[f"{v:.1f}" for v in synth_df["Score"]],textposition="outside"))
            fig.add_vline(x=70,line_dash="dot",line_color="#9CA3AF",
                annotation_text="Seuil 70",annotation_font_size=9)
            fig.update_layout(xaxis=dict(range=[50,90],title="Score /100"),height=400,**L)
            st.plotly_chart(fig,use_container_width=True)
        with col_b:
            # Format columns without Pandas Styler (not compatible with all Streamlit Cloud versions)
            tbl = synth_df[["Poste","Note","% Pos","% Nég","Score Rémun.","Score Charge","Score"]].copy()
            tbl["Note"]         = tbl["Note"].map("{:.2f}".format)
            tbl["% Pos"]        = tbl["% Pos"].map("{:.1f}%".format)
            tbl["% Nég"]        = tbl["% Nég"].map("{:.1f}%".format)
            tbl["Score Rémun."] = tbl["Score Rémun."].map(lambda x: f"{x:+.1f}" if not pd.isna(x) else "–")
            tbl["Score Charge"] = tbl["Score Charge"].map(lambda x: f"{x:+.1f}" if not pd.isna(x) else "–")
            tbl["Score"]        = tbl["Score"].map("{:.1f}".format)
            st.dataframe(tbl, height=400, use_container_width=True)

        sec("BUBBLE CHART — NOTE × RÉMUNÉRATION × CHARGE DE TRAVAIL")
        RP=SOUS_PRO_KEYS[9]; RC=SOUS_CON_KEYS[9]; CC=SOUS_CON_KEYS[8]
        bub=[]
        for pos in pos_avail:
            sub=dff_pos[dff_pos["pos_group"]==pos]
            if len(sub)<5: continue
            n=len(sub)
            rem=(sub[RP].sum()-sub[RC].sum())/n*100 if RP in sub.columns else 0
            chg=sub[CC].sum()/n*100 if CC in sub.columns else 0
            bub.append({"Poste":pos,"Note":sub["rating"].mean(),"Score Rémun.":rem,"% Charge":chg,"Nb":n})
        bdf=pd.DataFrame(bub)
        fig=px.scatter(bdf,x="Score Rémun.",y="Note",size=bdf["% Charge"].clip(lower=1),
            color="Poste",text="Poste",
            title="Taille bulle = % avis citant la charge en inconvénient",
            labels={"Score Rémun.":"Score net rémunération","Note":"Note /5"},
            color_discrete_map=POS_COLORS)
        fig.add_vline(x=0,line_dash="dot",line_color="#9CA3AF")
        fig.add_hline(y=bdf["Note"].mean(),line_dash="dot",line_color="#9CA3AF")
        fig.add_annotation(x=bdf["Score Rémun."].min()*0.85,y=bdf["Note"].min()*0.998,
            text="⚠️ Zone risque",showarrow=False,font=dict(color="#EF4444",size=10))
        fig.update_traces(textposition="top center",marker=dict(sizemin=8))
        fig.update_layout(height=440,showlegend=False,**L)
        st.plotly_chart(fig,use_container_width=True)

    with t4:
        sec("RECOMMANDATIONS PAR POSTE — INVESTISSEUR RH")
        recs_pos={
            "Stagiaire Audit":      ("🟢","Vitrine employeur","Note 4,18/5 · 85% positifs","Capitaliser pour le recrutement · Structurer la conversion CDI"),
            "Stagiaire (anonyme)":  ("🟢","Profil enthousiaste","Note 4,21/5 · 83% positifs","Améliorer l'accueil et le mentoring"),
            "Consultant Junior":    ("🟡","Friction financière","Rémunération −20,6","Réviser la grille salariale en entrée"),
            "Auditeur Junior":      ("🟡","Charge élevée","59% citent la charge en inconvénient","Maintenir l'avantage salarial relatif (+5,6)"),
            "Analyst / Transaction":("🔴","Priorité absolue","Score 60,0 · Rémun. −25,0 · 16,7% négatifs","Benchmarking salarial urgent vs marché"),
            "Consultant":           ("🟡","Profil mixte","Variable selon cabinet","Investiguer écarts inter-cabinets"),
            "Auditeur Senior":      ("🔴","Risque élevé","Score 66,6 · Charge 60,4%","Flexibilité horaire + revue salariale"),
            "Senior Consultant":    ("🟡","Autonomie insuffisante","Besoin délégation","Programmes de leadership et responsabilité"),
            "Manager":              ("🟢","Notes meilleures","70,8% positifs","Maintenir politiques de reconnaissance"),
            "Senior Manager":       ("🟡","Rémunération critique","Score rémun. −23,5","Surveiller la surcharge managériale"),
        }
        for i in range(0,len(pos_avail),4):
            cols=st.columns(4)
            for j,pos in enumerate(pos_avail[i:i+4]):
                if pos not in recs_pos: continue
                icon,signal,detail,action=recs_pos[pos]
                border={"🟢":"#10B981","🟡":"#F59E0B","🔴":"#EF4444"}[icon]
                with cols[j]:
                    st.markdown(f"""<div style='background:white;border:1px solid #E5E7EB;border-top:3px solid {border};
                        border-radius:10px;padding:14px;margin-bottom:10px;min-height:140px'>
                        <div style='font-size:.58rem;font-weight:800;color:#9CA3AF;text-transform:uppercase;letter-spacing:.08em'>{pos}</div>
                        <div style='font-size:.88rem;font-weight:700;color:#111827;margin:3px 0'>{icon} {signal}</div>
                        <div style='font-size:.67rem;color:#6B7280;margin-bottom:5px'>{detail}</div>
                        <div style='font-size:.65rem;color:{border};font-weight:700'>→ {action}</div>
                    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️   Points de Vigilance":
# ═══════════════════════════════════════════════════════════════════════════════

    st.markdown("""<div style='background:linear-gradient(135deg,#0F172A,#78350F);border-radius:14px;
        padding:24px 28px;margin-bottom:24px'>
        <div style='font-size:1.5rem;font-weight:800;color:white'>Points de Vigilance — Biais & Limites</div>
        <div style='font-size:.82rem;color:#FCD34D;margin-top:4px'>Éléments à intégrer avant toute décision d'investissement</div>
    </div>""", unsafe_allow_html=True)

    alert("⚠️ Ces analyses reposent sur des avis Glassdoor auto-déclarés. Elles constituent un <b>signal complémentaire</b>, non un substitut à une due diligence RH complète.","warn")
    st.markdown("<br>", unsafe_allow_html=True)

    col_a,col_b=st.columns(2)
    with col_a:
        sec("SUR-REPRÉSENTATIVITÉ DE KPMG")
        vol=dff["cabinet"].value_counts().reset_index(); vol.columns=["Cabinet","Nb avis"]
        vol["%"]=vol["Nb avis"]/vol["Nb avis"].sum()*100
        fig=go.Figure(go.Pie(labels=vol["Cabinet"],values=vol["Nb avis"],
            marker=dict(colors=[COLORS[c] for c in vol["Cabinet"]],
                        line=dict(color="white",width=2)),
            hole=0.6,textinfo="label+percent",textfont=dict(size=12)))
        fig.add_annotation(text=f"<b>{vol['Nb avis'].sum():,}</b><br>avis total",
            x=0.5,y=0.5,showarrow=False,font=dict(size=14,color="#111827"))
        fig.update_layout(height=280,showlegend=False,**L)
        st.plotly_chart(fig,use_container_width=True)
        alert("⚠️ <b>KPMG représente 62% des avis</b> (1 332/2 135). Les indicateurs globaux sont fortement influencés par KPMG. Deloitte (208 avis = 10%) a des estimations moins stables.","warn")

    with col_b:
        sec("BIAIS JUNIORS — ANCIENNETÉ SOUS-REPRÉSENTÉE")
        anc=dff.dropna(subset=["employment_duration"]).copy()
        anc["anc_bins"]=pd.cut(anc["employment_duration"],bins=[0,1,2,4,6,50],
            labels=["<1 an","1-2 ans","2-4 ans","4-6 ans",">6 ans"])
        anc_dist=anc.groupby("anc_bins",observed=True).size().reset_index(name="n")
        anc_dist["%"]=anc_dist["n"]/anc_dist["n"].sum()*100
        fig=px.bar(anc_dist,x="anc_bins",y="n",
            labels={"anc_bins":"Ancienneté","n":"Nb avis"},
            color="anc_bins",color_discrete_sequence=["#BAE6FD","#60A5FA","#3B82F6","#1D4ED8","#1E3A8A"])
        fig.add_vline(x=1.5,line_dash="dot",line_color="#EF4444",
            annotation_text="73% < 2 ans",annotation_font_color="#EF4444",annotation_font_size=10)
        fig.update_layout(showlegend=False,height=280,xaxis_title="",yaxis_title="Nb avis",**L)
        st.plotly_chart(fig,use_container_width=True)
        alert("⚠️ <b>73% des avis viennent d'employés avec moins de 2 ans</b>. Les profils seniors (>5 ans) et la Direction s'expriment peu sur Glassdoor.","warn")

    col_c,col_d=st.columns(2)
    with col_c:
        sec("REPRÉSENTATIVITÉ DES GRADES — DIRECTION SOUS-REPRÉSENTÉE")
        grd=dff_known["grade_clean"].value_counts().reindex(grade_avail).dropna()
        fig=go.Figure(go.Bar(x=grd.index,y=grd.values,
            marker_color=[GRADE_COLORS.get(g,"#9CA3AF") for g in grd.index],
            marker_line_width=0,
            text=[f"{v}<br>({v/len(dff_known)*100:.0f}%)" for v in grd.values],
            textposition="outside",textfont=dict(size=10)))
        fig.update_layout(height=270,yaxis_title="Nb avis",**L)
        st.plotly_chart(fig,use_container_width=True)
        alert("⚠️ <b>Direction = 19 avis seulement</b> (0,9% du total). Les Partners et Directeurs s'expriment rarement. Les conclusions sur ce grade sont peu robustes statistiquement.","warn")

    with col_d:
        sec("73% DES AVIS SANS DATE — LIMITE TEMPORELLE")
        has_date=dff["date_p"].notna()
        fig=go.Figure(go.Pie(labels=["Avec date","Sans date"],
            values=[has_date.sum(),len(dff)-has_date.sum()],
            marker=dict(colors=["#3B82F6","#E5E7EB"],line=dict(color="white",width=2)),
            hole=0.6,textfont=dict(size=12)))
        pct=has_date.mean()*100
        fig.add_annotation(text=f"<b>{pct:.0f}%</b><br>avec date",
            x=0.5,y=0.5,showarrow=False,font=dict(size=14,color="#111827"))
        fig.update_layout(height=270,**L,legend=dict(orientation="h",y=-0.1,font=dict(size=11)))
        st.plotly_chart(fig,use_container_width=True)
        alert("⚠️ <b>73% des avis sans date</b>. L'analyse temporelle (2019-2024) ne couvre que 27% du dataset. Les tendances récentes sont peu représentatives.","warn")

    sec("SYNTHÈSE DES BIAIS MÉTHODOLOGIQUES")
    biases=[
        ("📊","Biais de sélection Glassdoor","Les employés très satisfaits ET très mécontents écrivent plus. Les 'silencieux' (satisfaits sans motivation d'écrire) sont absents. Distribution bimodale typique (pics sur 1-2 et 4-5)."),
        ("🏢","KPMG sur-représenté (62%)","Les indicateurs globaux sont fortement influencés par KPMG. Deloitte (10% des avis) a une estimation statistiquement moins fiable. Comparer les cabinets avec prudence."),
        ("👶","Juniors sur-représentés (73% < 2 ans)","Les enjeux des profils seniors (5+ ans), des Managers expérimentés et de la Direction sont sous-capturés. Les analyses Grade sont à pondérer."),
        ("📅","73% sans données temporelles","L'analyse de tendance est limitée. Les conclusions sur l'évolution 2019-2024 reposent sur une minorité d'avis datés."),
        ("🌍","Avis en français uniquement","Ce dataset couvre les expériences françaises. Les équipes internationales (UK, US, Inde) ne sont pas capturées."),
        ("🔍","NLP par mots-clés","L'analyse thématique repose sur des mots-clés prédéfinis. Des nuances de sentiment au niveau de la phrase peuvent être manquées par cette approche."),
    ]
    for i in range(0,len(biases),3):
        cols=st.columns(3)
        for j,(icon,title,desc) in enumerate(biases[i:i+3]):
            with cols[j]:
                st.markdown(f"""<div style='background:white;border:1px solid #E5E7EB;border-radius:10px;
                    padding:16px;margin-bottom:10px;'>
                    <div style='font-size:1.4rem;margin-bottom:6px'>{icon}</div>
                    <div style='font-size:.78rem;font-weight:700;color:#111827;margin-bottom:4px'>{title}</div>
                    <div style='font-size:.7rem;color:#6B7280;line-height:1.5'>{desc}</div>
                </div>""", unsafe_allow_html=True)

    alert("💡 <b>Recommandation méthodologique :</b> Ces analyses constituent un signal d'alerte et de priorisation. Elles doivent être complétées par une due diligence RH terrain (entretiens, données de turnover réelles, politique de rémunération interne) avant toute décision d'investissement.","info")
