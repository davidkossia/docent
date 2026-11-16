#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { DocentStack } from "../lib/docent-stack";

const app = new App();
new DocentStack(app, "DocentStack");
