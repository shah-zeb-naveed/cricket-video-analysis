import azure.functions as func
import logging
from run_process import run_pipeline

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    youtube_url = req.params.get('youtube_url')
    reference_image_path = req.params.get('reference_image_path')
    output_video_path = req.params.get('output_video_path')
    
    if not youtube_url:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            youtube_url = req_body.get('youtube_url')
            reference_image_path = req_body.get('reference_image_path')
            output_video_path = req_body.get('output_video_path')
                        
    if output_video_path:
        run_pipeline([youtube_url], reference_image_path, output_video_path)
        return func.HttpResponse(f"Hello, {output_video_path}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully but no parameters were specified.",
             status_code=200
        )