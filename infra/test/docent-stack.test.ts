import * as fs from "fs";
import * as path from "path";
import { App } from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";
import { Code } from "aws-cdk-lib/aws-lambda";
import {
  DocentStack,
  DocentStackProps,
  KEY_PARAMETERS,
  LIMITS,
  QUOTA_KEY,
  QUOTA_TTL_ATTRIBUTE,
} from "../lib/docent-stack";

const ORIGIN = "https://docent.example.com";

function synth(overrides: Partial<DocentStackProps> = {}): Template {
  const app = new App();
  const stack = new DocentStack(app, "Test", {
    env: { region: "us-east-1", account: "123456789012" },
    code: Code.fromInline("def handler(event, context):\n    return None\n"),
    allowedOrigin: ORIGIN,
    dailyLimit: 150,
    setup: "section-800__pinecone-e5",
    ...overrides,
  });
  return Template.fromStack(stack);
}

const template = synth();

test("Lambda is Python 3.12 on arm64 with 1024 MB, a 29 s timeout and reserved concurrency 2", () => {
  template.resourceCountIs("AWS::Lambda::Function", 1);
  template.hasResourceProperties("AWS::Lambda::Function", {
    Runtime: "python3.12",
    Architectures: ["arm64"],
    Handler: "docent.lambda_handler.handler",
    MemorySize: 1024,
    Timeout: 29,
    ReservedConcurrentExecutions: 2,
  });
  expect(LIMITS.timeoutSeconds).toBeLessThan(30);
});

test("the stage throttles at rate 2 and burst 5", () => {
  template.resourceCountIs("AWS::ApiGatewayV2::Stage", 1);
  template.hasResourceProperties("AWS::ApiGatewayV2::Stage", {
    StageName: "$default",
    AutoDeploy: true,
    DefaultRouteSettings: { ThrottlingRateLimit: 2, ThrottlingBurstLimit: 5 },
  });
});

test("the API has one route, POST /ask", () => {
  template.resourceCountIs("AWS::ApiGatewayV2::Route", 1);
  template.hasResourceProperties("AWS::ApiGatewayV2::Route", { RouteKey: "POST /ask" });
});

test("CORS allows only the configured origin and POST", () => {
  template.hasResourceProperties("AWS::ApiGatewayV2::Api", {
    ProtocolType: "HTTP",
    CorsConfiguration: { AllowOrigins: [ORIGIN], AllowMethods: ["POST"], AllowHeaders: ["content-type"] },
  });
});

test("the quota table is on-demand with a TTL", () => {
  template.hasResourceProperties("AWS::DynamoDB::Table", {
    BillingMode: "PAY_PER_REQUEST",
    KeySchema: [{ AttributeName: QUOTA_KEY, KeyType: "HASH" }],
    TimeToLiveSpecification: { AttributeName: QUOTA_TTL_ATTRIBUTE, Enabled: true },
  });
});

test("IAM: no managed policies; logs write on the log group, UpdateItem on the table, GetParameters on the keys", () => {
  template.resourceCountIs("AWS::IAM::ManagedPolicy", 0);
  template.resourceCountIs("AWS::IAM::Role", 1);
  const role = Object.values(template.findResources("AWS::IAM::Role"))[0] as any;
  expect(role.Properties.ManagedPolicyArns).toBeUndefined();
  expect(role.Properties.Policies).toBeUndefined();
  expect(role.Properties.AssumeRolePolicyDocument.Statement).toEqual([
    { Action: "sts:AssumeRole", Effect: "Allow", Principal: { Service: "lambda.amazonaws.com" } },
  ]);

  const policies = template.findResources("AWS::IAM::Policy");
  const statements = Object.values(policies).flatMap((p: any) => p.Properties.PolicyDocument.Statement);
  expect(statements).toHaveLength(3);
  for (const s of statements) expect(s.Effect).toBe("Allow");

  const logs = statements.find((s: any) => JSON.stringify(s.Action).includes("logs:"));
  expect(logs.Action).toEqual(["logs:CreateLogStream", "logs:PutLogEvents"]);
  const [logGroupId] = Object.keys(template.findResources("AWS::Logs::LogGroup"));
  expect(logs.Resource).toEqual({ "Fn::GetAtt": [logGroupId, "Arn"] });

  const dynamo = statements.find((s: any) => JSON.stringify(s.Action).includes("dynamodb"));
  expect(dynamo.Action).toBe("dynamodb:UpdateItem");
  const [tableId] = Object.keys(template.findResources("AWS::DynamoDB::Table"));
  expect(dynamo.Resource).toEqual([{ "Fn::GetAtt": [tableId, "Arn"] }]);

  const ssm = statements.find((s: any) => JSON.stringify(s.Action).includes("ssm"));
  expect(ssm.Action).toBe("ssm:GetParameters");
  const arns = ssm.Resource.map((r: any) => r["Fn::Join"][1].map((p: any) => (typeof p === "string" ? p : "<partition>")).join(""));
  expect(arns).toEqual([
    "arn:<partition>:ssm:us-east-1:123456789012:parameter/docent/anthropic_api_key",
    "arn:<partition>:ssm:us-east-1:123456789012:parameter/docent/pinecone_api_key",
  ]);
});

test("no IAM resource anywhere is \"*\"", () => {
  const iam = { ...template.findResources("AWS::IAM::Role"), ...template.findResources("AWS::IAM::Policy"),
    ...template.findResources("AWS::IAM::ManagedPolicy") };
  const json = JSON.stringify(iam);
  expect(json).not.toMatch(/"Resource":"\*"|"Resource":\["\*"/);
  expect(json).not.toContain('"Action":"*"');
  expect(json).not.toContain("arn:aws:iam::aws:policy");
});

test("asynchronous invokes are never retried", () => {
  template.hasResourceProperties("AWS::Lambda::EventInvokeConfig", { MaximumRetryAttempts: 0 });
});

test("no secret in the template: no key values, no key variables, no secret resources", () => {
  const json = JSON.stringify(template.toJSON());
  expect(json).not.toMatch(/sk-ant-|pcsk_|AKIA[0-9A-Z]{16}/);
  template.hasResourceProperties("AWS::Lambda::Function", {
    Environment: { Variables: { QUOTA_TABLE: Match.anyValue(), DAILY_LIMIT: "150", DOCENT_SETUP: "section-800__pinecone-e5" } },
  });
  const fn = Object.values(template.findResources("AWS::Lambda::Function"))[0] as any;
  expect(Object.keys(fn.Properties.Environment.Variables).sort()).toEqual(["DAILY_LIMIT", "DOCENT_SETUP", "QUOTA_TABLE"]);
  template.resourceCountIs("AWS::SSM::Parameter", 0);
  template.resourceCountIs("AWS::SecretsManager::Secret", 0);
});

test("the API URL is an output", () => {
  template.hasOutput("ApiUrl", {});
});

test.each([
  "*",
  "",
  undefined,
  "https://docent.example.com/",
  "docent.example.com",
  "ftp://docent.example.com",
  "http://docent.example.com",
])("rejects allowedOrigin %s", (allowedOrigin) => {
  expect(() => synth({ allowedOrigin: allowedOrigin as string })).toThrow(/allowedOrigin/);
});

test.each(["https://docent.example.com", "http://localhost:5173", "http://127.0.0.1:5173"])(
  "accepts allowedOrigin %s",
  (allowedOrigin) => expect(() => synth({ allowedOrigin })).not.toThrow(),
);

test.each([-1, 1.5, NaN])("rejects dailyLimit %s", (dailyLimit) => {
  expect(() => synth({ dailyLimit })).toThrow(/dailyLimit/);
});

test("names match the Python handler and quota code", () => {
  const backend = path.resolve(__dirname, "../../backend/docent");
  const handler = fs.readFileSync(path.join(backend, "lambda_handler.py"), "utf8");
  const quota = fs.readFileSync(path.join(backend, "quota.py"), "utf8");
  for (const name of KEY_PARAMETERS) expect(handler).toContain(`"${name}"`);
  expect(handler).toContain('os.environ["QUOTA_TABLE"]');
  expect(quota).toContain(`KEY = "${QUOTA_KEY}"`);
  expect(quota).toContain(`EXPIRES = "${QUOTA_TTL_ATTRIBUTE}"`);
  expect(quota).toContain('"DAILY_LIMIT"');
});
