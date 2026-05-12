echo "🚀 CMMD pipeline starting..."

echo "📦 Note: CMMD download is handled via TCIA Data Retriever"
echo "👉 Please launch TCIA tool and select CMMD dataset"

python src/download/download_cmmd.py --max_cases 20
