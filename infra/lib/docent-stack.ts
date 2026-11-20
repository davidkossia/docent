import { CfnOutput, Duration, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import { CorsHttpMethod, HttpApi, HttpMethod } from "aws-cdk-lib/aws-apigatewayv2";
import { HttpLambdaIntegration } from "aws-cdk-lib/aws-apigatewayv2-integrations";
import { AttributeType, BillingMode, Table } from "aws-cdk-lib/aws-dynamodb";
import { PolicyStatement, Role, ServicePrincipal } from "aws-cdk-lib/aws-iam";
import { Architecture, Code, Function, Runtime } from "aws-cdk-lib/aws-lambda";
import { LogGroup, RetentionDays } from "aws-cdk-lib/aws-logs";
import { Construct } from "constructs";

// Names shared with backend/docent/lambda_handler.py and quota.py. The CDK test checks they match.
export const KEY_PARAMETERS = ["/docent/anthropic_api_key", "/docent/pinecone_api_key"];
export const QUOTA_KEY = "day";
export const QUOTA_TTL_ATTRIBUTE = "expires_at";

export const LIMITS = {
  memoryMb: 1024,
  timeoutSeconds: 29, // API Gateway HTTP APIs cut requests at 30 seconds
  reservedConcurrency: 2,
  throttleRate: 2, // requests per second, whole stage
  throttleBurst: 5,
};

export interface DocentStackProps extends StackProps {
  /** The Lambda bundle (backend/build/lambda from infra/scripts/build_lambda.sh). */
  readonly code: Code;
  /** The one origin allowed by CORS, such as https://docent.example.com. */
  readonly allowedOrigin: string;
  /** Questions per UTC day that may reach Claude. */
  readonly dailyLimit: number;
  /** `<strategy>__<embedder>`: the Pinecone namespace to search. */
  readonly setup: string;
}

export class DocentStack extends Stack {
  constructor(scope: Construct, id: string, props: DocentStackProps) {
    super(scope, id, props);
    const origin = validOrigin(props.allowedOrigin);
    if (!Number.isInteger(props.dailyLimit) || props.dailyLimit < 0) {
      throw new Error(`dailyLimit must be a whole number of 0 or more, not ${props.dailyLimit}`);
    }
    if (!/^[a-z0-9-]+__[a-z0-9-]+$/.test(props.setup ?? "")) {
      throw new Error(`setup must look like <strategy>__<embedder>, not ${props.setup}`);
    }

    const quota = new Table(this, "Quota", {
      partitionKey: { name: QUOTA_KEY, type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: QUOTA_TTL_ATTRIBUTE,
      removalPolicy: RemovalPolicy.DESTROY, // a daily counter; nothing to keep after cdk destroy
    });

    const logs = new LogGroup(this, "AskLogs", {
      retention: RetentionDays.TWO_WEEKS,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // No managed policies: the role can write to this log group, update the quota table, and read
    // the two key parameters, and nothing else.
    const role = new Role(this, "AskRole", { assumedBy: new ServicePrincipal("lambda.amazonaws.com") });
    logs.grantWrite(role);

    // Keys are read from SSM at cold start. Only these three non-secret values go in the environment.
    const fn = new Function(this, "Ask", {
      runtime: Runtime.PYTHON_3_12,
      architecture: Architecture.ARM_64,
      handler: "docent.lambda_handler.handler",
      code: props.code,
      memorySize: LIMITS.memoryMb,
      timeout: Duration.seconds(LIMITS.timeoutSeconds),
      reservedConcurrentExecutions: LIMITS.reservedConcurrency,
      retryAttempts: 0, // never re-run a question on an asynchronous invoke
      role,
      logGroup: logs,
      environment: {
        QUOTA_TABLE: quota.tableName,
        DAILY_LIMIT: String(props.dailyLimit),
        DOCENT_SETUP: props.setup,
      },
    });

    quota.grant(fn, "dynamodb:UpdateItem");
    fn.addToRolePolicy(
      new PolicyStatement({
        actions: ["ssm:GetParameters"],
        resources: KEY_PARAMETERS.map((name) =>
          this.formatArn({ service: "ssm", resource: "parameter", resourceName: name.replace(/^\//, "") }),
        ),
      }),
    );

    const api = new HttpApi(this, "HttpApi", {
      createDefaultStage: false,
      corsPreflight: {
        allowOrigins: [origin],
        allowMethods: [CorsHttpMethod.POST],
        allowHeaders: ["content-type"],
        maxAge: Duration.days(1),
      },
    });
    api.addStage("Stage", {
      stageName: "$default",
      autoDeploy: true,
      throttle: { rateLimit: LIMITS.throttleRate, burstLimit: LIMITS.throttleBurst },
    });
    api.addRoutes({
      path: "/ask",
      methods: [HttpMethod.POST],
      integration: new HttpLambdaIntegration("AskIntegration", fn),
    });

    new CfnOutput(this, "ApiUrl", {
      value: api.apiEndpoint,
      description: "Base URL for the UI's VITE_API_URL; the route is POST /ask",
    });
  }
}

const LOCAL_HOSTS = ["localhost", "127.0.0.1", "[::1]"];

function validOrigin(value: string | undefined): string {
  if (!value) {
    throw new Error("allowedOrigin is required: pass -c allowedOrigin=https://<the UI's origin>");
  }
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`allowedOrigin must be an origin such as https://docent.example.com, not ${value}`);
  }
  if (url.origin !== value) {
    throw new Error(`allowedOrigin must be a bare origin (scheme, host, port), not ${value}`);
  }
  const local = LOCAL_HOSTS.includes(url.hostname);
  if (!(url.protocol === "https:" || (url.protocol === "http:" && local))) {
    throw new Error(`allowedOrigin must use https (http only for localhost), not ${value}`);
  }
  return value;
}
