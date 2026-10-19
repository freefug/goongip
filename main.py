import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 페이지 및 브라우저 탭 설정
st.set_page_config(
    page_title="영화 유형 나누기", page_icon="🎬", layout="wide"
)

st.title("🎬 영화 유형 나누기")


# 데이터 로드 및 전처리 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")

    # 필수 열에 대한 결측치 및 조건 처리
    req_cols = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
    df_clean = df.dropna(subset=req_cols).copy()
    df_clean = df_clean[df_clean["first_week_audi"] > 0].copy()

    # 원본 수치 보존 (표시 및 평균 계산용)
    df_clean["long_run_index"] = (
        df_clean["total_audi"] / df_clean["first_week_audi"]
    )
    df_clean["long_run_index"] = df_clean["long_run_index"].clip(upper=20)

    # 파생 변수 (군집화 및 산점도용 변환 속성)
    df_clean["log_first_scrn"] = np.log10(
        df_clean["first_scrn"].replace(0, np.nan)
    )
    df_clean["log_total_audi"] = np.log10(
        df_clean["total_audi"].replace(0, np.nan)
    )

    # 변환 후 발생할 수 있는 결측치 제거
    df_clean = df_clean.dropna(
        subset=["log_first_scrn", "log_total_audi"]
    ).reset_index(drop=True)

    return df, df_clean


raw_df, df = load_data()

# 전체 편수 및 묶은 편수 한 줄 표시
st.write(
    f"**전체 영화 편수:** {len(raw_df):,}편  |  **분석 대상 영화 편수:** {len(df):,}편"
)

st.markdown("---")

# 속성 매핑 사전
feature_dict = {
    "스크린 수(로그)": "log_first_scrn",
    "누적 관객 수(로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}

# 1. 묶는 데 사용할 속성 및 묶음 수 선택 UI
col_opt1, col_opt2 = st.columns([3, 2])

with col_opt1:
    selected_features_labels = st.multiselect(
        "묶는 데 사용할 속성을 선택하세요 (최소 2개 이상):",
        options=list(feature_dict.keys()),
        default=list(feature_dict.keys()),
    )

with col_opt2:
    n_clusters = st.slider(
        "묶음 수(k)를 선택하세요:", min_value=2, max_value=7, value=3, step=1
    )

if len(selected_features_labels) < 2:
    st.warning("속성을 2개 이상 선택해 주세요.")
    st.stop()

selected_features = [feature_dict[lbl] for lbl in selected_features_labels]

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_features])

# 선택한 K-Means 클러스터링
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["cluster_raw"] = kmeans.fit_predict(X_scaled)

# 누적 관객 수 평균에 따라 ㉮~㉴ 기호 매핑
all_cluster_labels = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]
cluster_audi_mean = (
    df.groupby("cluster_raw")["total_audi"].mean().sort_values(ascending=False)
)
cluster_mapping = {}
for idx, cluster_id in enumerate(cluster_audi_mean.index):
    cluster_mapping[cluster_id] = all_cluster_labels[idx]

df["cluster"] = df["cluster_raw"].map(cluster_mapping)
active_labels = all_cluster_labels[:n_clusters]

st.markdown("---")

# 2차원 산점도
st.subheader("📌 2차원 산점도")
col2d1, col2d2 = st.columns(2)
with col2d1:
    x_2d_label = st.selectbox(
        "2D 가로축(X):",
        options=selected_features_labels,
        index=0,
        key="2d_x",
    )
with col2d2:
    y_2d_label = st.selectbox(
        "2D 세로축(Y):",
        options=selected_features_labels,
        index=min(1, len(selected_features_labels) - 1),
        key="2d_y",
    )

fig_2d = px.scatter(
    df,
    x=feature_dict[x_2d_label],
    y=feature_dict[y_2d_label],
    color="cluster",
    hover_name="movieNm",
    labels={
        feature_dict[x_2d_label]: x_2d_label,
        feature_dict[y_2d_label]: y_2d_label,
        "cluster": "묶음",
    },
    title=f"2D 산점도: {x_2d_label} vs {y_2d_label}",
    category_orders={"cluster": active_labels},
)
st.plotly_chart(fig_2d, use_container_width=True)

# 3차원 산점도
st.subheader("📌 3차원 산점도")
if len(selected_features_labels) < 3:
    st.info("3차원 산점도를 그리려면 속성을 3개 이상 선택해야 합니다.")
else:
    col3d1, col3d2, col3d3 = st.columns(3)
    with col3d1:
        x_3d_label = st.selectbox(
            "3D X축:",
            options=selected_features_labels,
            index=0,
            key="3d_x",
        )
    with col3d2:
        y_3d_label = st.selectbox(
            "3D Y축:",
            options=selected_features_labels,
            index=1,
            key="3d_y",
        )
    with col3d3:
        z_3d_label = st.selectbox(
            "3D Z축:",
            options=selected_features_labels,
            index=min(2, len(selected_features_labels) - 1),
            key="3d_z",
        )

    fig_3d = px.scatter_3d(
        df,
        x=feature_dict[x_3d_label],
        y=feature_dict[y_3d_label],
        z=feature_dict[z_3d_label],
        color="cluster",
        hover_name="movieNm",
        labels={
            feature_dict[x_3d_label]: x_3d_label,
            feature_dict[y_3d_label]: y_3d_label,
            feature_dict[z_3d_label]: z_3d_label,
            "cluster": "묶음",
        },
        title=f"3D 산점도: {x_3d_label} vs {y_3d_label} vs {z_3d_label}",
        category_orders={"cluster": active_labels},
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# 묶음별 요약 통계량 (원래 단위 평균)
st.subheader("📊 묶음별 평균 및 편수")

summary_df = (
    df.groupby("cluster")
    .agg(
        편수=("movieNm", "count"),
        스크린_수_평균=("first_scrn", "mean"),
        누적_관객_평균=("total_audi", "mean"),
        _10위권_일수_평균=("days_in_top10", "mean"),
        롱런_지수_평균=("long_run_index", "mean"),
    )
    .reindex(active_labels)
)

summary_df.columns = [
    "편수",
    "스크린 수 (개)",
    "누적 관객 (명)",
    "10위권 일수 (일)",
    "롱런 지수",
]
summary_df["편수"] = summary_df["편수"].map("{:,}편".format)
summary_df["스크린 수 (개)"] = summary_df["스크린 수 (개)"].map(
    "{:,.1f}".format
)
summary_df["누적 관객 (명)"] = summary_df["누적 관객 (명)"].map(
    "{:,.0f}".format
)
summary_df["10위권 일수 (일)"] = summary_df["10위권 일수 (일)"].map(
    "{:,.1f}".format
)
summary_df["롱런 지수"] = summary_df["롱런 지수"].map("{:,.2f}".format)

st.dataframe(summary_df, use_container_width=True)

# 묶음별 누적 관객 수가 많은 영화 Top 5
st.subheader("🏆 묶음별 누적 관객 수 Top 5 영화")
top_cols = st.columns(n_clusters)

for idx, label in enumerate(active_labels):
    with top_cols[idx]:
        st.markdown(f"### 묶음 **{label}**")
        top5 = (
            df[df["cluster"] == label]
            .sort_values(by="total_audi", ascending=False)
            .head(5)
        )
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            st.write(f"**{i}. {row['movieNm']}** ({row['total_audi']:,}명)")

st.markdown("---")

# 엘보우 기법 (Elbow Method) 및 실루엣 점수 평가
st.subheader("📉 적정 묶음 수 평가 (Elbow Method & Silhouette Score)")

# k=1~7 inertia 계산
k_range = list(range(1, 8))
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Inertia 꺾은선 그래프 생성
fig_elbow = go.Figure()
fig_elbow.add_trace(
    go.Scatter(
        x=k_range,
        y=inertias,
        mode="lines+markers",
        name="Inertia",
        marker=dict(size=8),
    )
)

# 현재 선택된 k 위치에 세로선 표시
fig_elbow.add_vline(
    x=n_clusters,
    line_dash="dash",
    line_color="red",
    annotation_text=f"현재 선택 (k={n_clusters})",
    annotation_position="top right",
)

fig_elbow.update_layout(
    title="묶음 수(k)에 따른 Inertia (거리 제곱합) 변화",
    xaxis_title="묶음 수 (k)",
    yaxis_title="Inertia (중심과의 거리 제곱합)",
    xaxis=dict(tickmode="linear", tick0=1, dtick=1),
)
st.plotly_chart(fig_elbow, use_container_width=True)

# Inertia 및 전단계 대비 감소량 표 작성
inertia_diffs = [""]  # k=1 일때 비교값 없음
for i in range(1, len(inertias)):
    diff = inertias[i - 1] - inertias[i]
    inertia_diffs.append(f"{diff:,.2f}")

elbow_table_df = pd.DataFrame(
    {
        "묶음 수 (k)": k_range,
        "Inertia (거리 제곱합)": [f"{val:,.2f}" for val in inertias],
        "전단계 대비 감소량": inertia_diffs,
    }
)

st.dataframe(elbow_table_df, use_container_width=True)

# 현재 선택된 k에 대한 실루엣 점수 계산 및 안내
current_silhouette = silhouette_score(X_scaled, kmeans.labels_)
st.info(
    f"💡 **현재 선택한 묶음 수({n_clusters})의 실루엣 점수 (Silhouette Score):** `{current_silhouette:.4f}`\n\n"
    "*(실루엣 점수는 -1에서 1 사이의 값을 가지며, 1에 가까울수록 군집이 서로 명확하게 구분됨을 의미합니다.)*"
)
