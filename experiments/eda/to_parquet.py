import polars as pl, time
base="student_resource/dataset"
for split in ["train","test"]:
    for s in ["source1","source2","source3"]+(["ground_truth"] if split=="train" else []):
        t=time.time()
        df=pl.read_csv(f"{base}/{split}/{split}_{s}.tsv",separator="\t",quote_char=None,infer_schema_length=0)
        df.write_parquet(f"work/data/{split}_{s}.parquet")
        print(split,s,df.shape,df.columns,f"{time.time()-t:.1f}s",flush=True)
