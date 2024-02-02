# Callback to update the graph based on the input and dropdown selections
@app.callback(
    Output('prediction-plot', 'figure'),
    [Input('defect-desc-input', 'value'),
     Input('job-detail-input', 'value'),
     Input('job-sum-input', 'value'),
     Input('job-head-input', 'value'),
     Input('categorical-dropdown-shipname', 'value'),
     Input('categorical-dropdown-refitcode', 'value'),
     Input('categorical-dropdown-offloaded', 'value'),
     Input('categorical-dropdown-equipment', 'value'),
     Input('categorical-dropdown-qcremark', 'value')]
)
def update_graph(defect_desc, job_detail, job_sum, job_head, selected_shipname, selected_refitcode, selected_offloaded, selected_equipment, selected_qcremark):
    if defect_desc and job_detail and job_sum and job_head and selected_shipname and selected_refitcode and selected_offloaded and selected_equipment and selected_qcremark:
        # Create a DataFrame for the input features
        input_data = pd.DataFrame({
            'DEFECT_DESC': [defect_desc],
            'JOBDETAIL': [job_detail],
            'JOBSUM': [job_sum],
            'JOBHEAD': [job_head],
            'STR_SHIPNAME': [selected_shipname],
            'STR_REFIT_CODE': [selected_refitcode],
            'FLG_OFFLOADED': [selected_offloaded],
            'EQUIPMENT_NAME': [selected_equipment],
            'WI_QC_REMARK': [selected_qcremark]
        })

        # Make predictions for the transformed input data
        prediction = model.predict(input_data)

        # Generate a normal distribution curve for target column "EMD"
        mu, sigma = 500, 150 # mean and standard deviation
        s = np.random.normal(mu, sigma, 1000)

        # Create a histogram using Plotly
        fig = go.Figure(data=[go.Histogram(x=s, nbinsx=30, name='Actual Data')])
        fig.add_trace(go.Scatter(x=[prediction[0], prediction[0]], y=[0, 0.0025], mode='lines', name='Predicted Value'))
        fig.add_trace(go.Scatter(x=[prediction[0]], y=[0.0025], mode='markers', marker=dict(size=10, color='red'), name='Predicted Point'))

        fig.update_layout(
            title_text='Normal Distribution Curve for EMD with Predicted Value',
            xaxis_title_text='EMD',
            yaxis_title_text='Density',
            bargap=0.2,
            bargroupgap=0.1
        )

        return fig
    else:
        # If no input is provided, return an empty graph
        return px.scatter()

# Run the app
if __name__ == '__main__':
    app.run_server(host='192.168.0.0', port=8050, debug=True)
