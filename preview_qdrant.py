from qdrant_client import QdrantClient
from ingest_data import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME

def fetch_all_points(qdrant_client, collection_name, batch_size=100):
    offset = None
    total_points = 0

    while True:
        results, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )

        if not results:
            break

        for point in results:
            print(f"ID: {point.id}")
            print(f"Payload: {point.payload}")
            if point.vector:
                print(f"Vector (dim {len(point.vector)}): {point.vector[:5]}...")  # Show first 5 dims
            else:
                print("Vector: ❌ Not Found")
            print("="*70)
            total_points += 1

        if not next_offset:
            break
        offset = next_offset

    print(f"\n✅ Total points fetched: {total_points}")

if __name__ == "__main__":
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        fetch_all_points(client, COLLECTION_NAME, batch_size=100)
    except Exception as e:
        print(f"Error: {e}")