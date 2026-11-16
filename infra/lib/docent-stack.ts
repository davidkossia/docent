import { Stack, StackProps } from "aws-cdk-lib";
import { Construct } from "constructs";

// Empty on purpose. The Lambda and API Gateway resources land in the API milestone (see PLAN.md).
export class DocentStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);
  }
}
