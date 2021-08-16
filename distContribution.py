def distContribution(sc, df, feature_cols, model):
    """
    Calculates contributions of each feature based on SHAP values for the groups of observations for an xgboost model.
    
    Arguments:
        sc: SparkContext.
        df (pyspark.sql.DataFrame): Input data frame containing feature_cols.
        feature_cols (list[str]): List of feature columns in df.
        model_path (str): Path to model on Spark driver (or dbfs)
    Returns:
        df (pyspark.sql.DataFrame): Output data frame with grouped contributions
    """
    model.feature_names = feature_cols
    #broadcasting the model to all the nodes
    clf = sc.broadcast(model)
    

    @F.pandas_udf(schema,functionType=F.PandasUDFType.GROUPED_MAP)
    def calculateContributions(df):      
      import xgboost as xgb
      import pandas as pd
      import numpy as np
      
      dfx = df[feature_cols]
      gr = df['key'].iloc[0]
      dtm = xgb.DMatrix(dfx, label=df.target)
      dtm.feature_names = model.feature_names
      
      shap_values = clf.value.predict(dtm, pred_contribs=True)
      shapDf = pd.DataFrame(shap_values[:,:-1], columns=feature_cols )
      total_cont = pd.DataFrame(shapDf.sum(0)).reset_index()
      total_cont['key'] = gr
      total_cont.columns=["feature", "total_contribution",'key']
      
      pos = total_cont[total_cont.total_contribution>0]
      neg = total_cont[total_cont.total_contribution<0]
      pos['perc_cont'] = (pos['total_contribution']/pos['total_contribution'].sum())*100
      neg['perc_cont'] = (neg['total_contribution']/neg['total_contribution'].sum())*-100

      pos = pos.sort_values("total_contribution", ascending=False).head(6)
      neg = neg.sort_values("total_contribution").head(6)

      df_con = pd.concat([pos,neg])
      df2 = df_con.iloc[df_con.perc_cont.abs().argsort()[::-1]]
      df2 = df2[np.abs(df2["perc_cont"]) > 2]
      return df2[['feature',	'perc_cont','key']]
      

    grouped_shap = df.groupBy("key").apply(calculateContributions)
    return grouped_shap
 
