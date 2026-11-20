#!/usr/bin/env node
import * as fs from "fs";
import * as path from "path";
import { App } from "aws-cdk-lib";
import { Code } from "aws-cdk-lib/aws-lambda";
import { DocentStack } from "../lib/docent-stack";

const app = new App();
const bundle = path.resolve(__dirname, "../../backend/build/lambda");
if (!fs.existsSync(path.join(bundle, "docent", "lambda_handler.py"))) {
  throw new Error(`No Lambda bundle in ${bundle}. Run infra/scripts/build_lambda.sh first.`);
}

// Defaults live in cdk.json. allowedOrigin has none: pass -c allowedOrigin=https://<the UI's origin>
new DocentStack(app, "DocentStack", {
  env: { region: app.node.tryGetContext("region") },
  code: Code.fromAsset(bundle),
  allowedOrigin: app.node.tryGetContext("allowedOrigin"),
  dailyLimit: Number(app.node.tryGetContext("dailyLimit")),
  setup: app.node.tryGetContext("setup"),
});
