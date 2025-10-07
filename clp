

Search⌘K
Chat
⌘J
Voice
Imagine
Projects
History
Today
Yesterday
This Week
October
September
we have 2 spark tables. One stores customer ids, country and the risk level of country they had transacted with plus transaction date and total amount.
the other one has one row for each customer which stores the name of the countries the customer supposed to transact. there is a filed for that like this:
China; Turkey; 
I want to have list of customers sorted desc by amount so that they have transaction with countries outside of the list of expected in python/pyspark. what to do?
Thought for 20s
python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, trim, transform, array_contains, sum as pyspark_sum

# Initialize Spark session (assuming it's already set up in your environment)
spark = SparkSession.builder.appName("UnexpectedTransactions").getOrCreate()

# Load the tables as DataFrames (replace with your actual table names)
df_transactions = spark.table("transactions")  # Columns: customer_id, country, risk_level, transaction_date, amount
df_expected = spark.table("expected_countries")  # Columns: customer_id, expected_countries (e.g., "China; Turkey;")

# Process expected_countries: split by ';', trim spaces, and filter out empty strings
df_expected = df_expected.withColumn(
    "expected_array",
    transform(
        split(col("expected_countries"), ";"),
        lambda x: trim(x)
    )
).withColumn(
    "expected_array",
    pyspark.functions.array_remove(col("expected_array"), "")  # Remove empty strings if any
)

# Join transactions with expected countries on customer_id
df_joined = df_transactions.join(df_expected, on="customer_id", how="inner")

# Filter transactions where country is NOT in the expected_array
df_unexpected = df_joined.filter(~array_contains(col("expected_array"), col("country")))

# Group by customer_id and sum the amounts for unexpected transactions
df_result = df_unexpected.groupBy("customer_id").agg(
    pyspark_sum("amount").alias("total_unexpected_amount")
)

# Sort descending by total_unexpected_amount
df_result = df_result.orderBy(col("total_unexpected_amount").desc())

# Show the result (or collect to list if needed)
df_result.show()
2.7s


Upgrade to SuperGrok
New conversation - Grok
