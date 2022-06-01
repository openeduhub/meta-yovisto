import json
import sys

import cherrypy
from predict import Recommender

r = None


class WebService(object):
    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def recommend(self):
        data = cherrypy.request.json
        print(data)
        output = r.run(data["doc"])
        return json.dumps(output)


if __name__ == "__main__":

    modelFile = sys.argv[1]
    idFile = sys.argv[2]

    r = Recommender(modelFile, idFile)

    config = {"server.socket_host": "0.0.0.0"}
    cherrypy.config.update(config)
    cherrypy.quickstart(WebService())
