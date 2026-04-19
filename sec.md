# 总体大纲

------

## 第 0 部分：先建立 assessment 的总框架

### 这一部分讲什么

先不讲术语，先讲你到底在做什么：
 **判断一个 project 是否符合 rules，本质上是在检查什么。**

### 你会学到

- assessment 到底不是在干什么
- 什么叫“有技术”不等于“符合规则”
- 什么叫“规则、架构、实现、配置、证据”这 5 层
- 做 assessment 的基本思路：
   **规则要求 → 系统哪一层实现 → 去哪里找证据 → 怎么下结论**

### 这一部分会用到哪些 rules

这部分是总框架，适用于所有 rules。

------

## 第 1 部分：系统里到底有哪些“角色”和“参与方”

### 这一部分讲什么

先搞清楚系统里“谁是谁”，因为后面所有规则都建立在这个基础上。

### 会讲的核心概念

- User
- Client
- Public Client
- Confidential Client
- API Provider
- API Gateway
- Business Service Provider
- OpenID Provider / Authorization Server
- WebSSO
- Backend for Frontend (BFF)

### 在哪个环节会用到

当你看架构图、代码调用链、网关流量路径时，你要先知道：

- 哪个是前端
- 哪个是后端
- 哪个是网关
- 哪个是认证中心
- 哪个是在代表用户调用
- 哪个是在代表应用自己调用

### 涉及哪些 rules

- API0-R2
- API0-R3
- API2-R1
- API2-R7
- API1-R1
- API3-R3

------

## 第 2 部分：认证和授权，到底有什么区别

### 这一部分讲什么

这是最重要的基础之一。
 你必须彻底分清：

- Authentication：你是谁
- Authorization：你能做什么

### 会讲的核心概念

- Authentication
- Authorization
- Identity
- Permission
- Role
- Scope
- Need-to-know
- Least privilege
- Coarse-grained authorization
- Fine-grained authorization

### 在哪个环节会用到

当你判断一个项目是不是：

- 只是“能登录”
- 还是“真正限制了权限”
- 是不是“给多了数据”
- 是不是“权限开太大”

### 涉及哪些 rules

- API3-R1
- API3-R4
- API1-R1
- API5-R1
- API3-R2
- API3-R3
- API2-R9

------

## 第 3 部分：会话、登录态、SSO，到底是什么

### 这一部分讲什么

很多项目“看起来能登录”，但 session 管理其实有问题。
 这部分讲登录之后那段“持续有效状态”是怎么回事。

### 会讲的核心概念

- Session
- Cookie
- SSO
- Session timeout
- Idle timeout
- Absolute timeout
- Logout
- Session invalidation
- Concurrent sessions

### 在哪个环节会用到

当你判断：

- logout 是否真的退出
- 用户长期不操作后会不会失效
- 改密码后旧会话会不会被踢掉
- WebSSO 和 access token 生命周期是否协调

### 涉及哪些 rules

- API2-R4
- API2-R8
- API0-R2

------

## 第 4 部分：OAuth2 / OIDC 是什么，为什么它们总出现

### 这一部分讲什么

这部分讲“现代 API 为什么几乎都绕不开 OAuth2 / OIDC”。

### 会讲的核心概念

- OAuth 2.0
- OpenID Connect (OIDC)
- Authorization Server
- Authentication vs delegated authorization
- Why OAuth2 and OIDC are not the same thing
- Browser-based app
- Native app
- Service-to-service app

### 在哪个环节会用到

当你看到规则里说：

- 必须使用 OIDC / OAuth2
- 必须验证 flow
- 必须看 token
- 必须看 PKCE
   你就知道它在说哪一层。

### 涉及哪些 rules

- API0-R2
- API2-R1
- API2-R6
- API2-R7

------

## 第 5 部分：Grant Flow——token 到底是怎么拿到的

### 这一部分讲什么

这是“客户端如何获得 token”的部分。
 很多规则不是在看有没有 token，而是在看：**拿 token 的方式对不对。**

### 会讲的核心概念

- Grant flow
- Authorization Code Grant
- PKCE
- Client Credentials
- Device Code Grant
- Token Exchange
- Redirect URI
- Public vs confidential client in different flows

### 在哪个环节会用到

当你看：

- 前端登录流程
- BFF 登录流程
- 后端服务调 APIM 的拿 token 代码
- partner app 或 mobile app 的接入方式

### 涉及哪些 rules

- API2-R1
- API2-R6
- API0-R2
- API3-R4

------

## 第 6 部分：Token 全家桶——每种 token 分别干什么

### 这一部分讲什么

这是你现在最容易乱的地方。
 我们会把所有 token 拆开，讲清楚各自的用途和区别。

### 会讲的核心概念

- Token
- Access Token
- ID Token
- Refresh Token
- API Key
- JWT authentication token
- Authorization Code
- Device Code
- User Code
- Bearer token

### 在哪个环节会用到

当你读代码、日志、Swagger、认证流程图时，你能分清：

- 哪个 token 是给 API 用的
- 哪个 token 是给客户端看的
- 哪个 token 是用来续命的
- 哪个只是中间凭证
- 哪个只是应用标识

### 涉及哪些 rules

- API2-R2
- API2-R3
- API2-R5
- API2-R6
- API2-R7
- API2-R8
- API6-R1

------

## 第 7 部分：Token 里面的字段——claims 到底是什么

### 这一部分讲什么

很多规则会要求检查 token validity，但你必须知道“检查什么”。

### 会讲的核心概念

- Claim
- iss
- sub
- aud
- exp
- nbf
- kid
- at_hash
- acr
- amr
- auth_time

### 在哪个环节会用到

当你判断：

- token 是不是合法签发的
- token 是不是给对的 API 用的
- token 是不是已经过期
- 用户认证强度够不够
- client authentication 是否合理

### 涉及哪些 rules

- API2-R3
- API2-R5
- API2-R7
- API2-R9
- API6-R1

------

## 第 8 部分：client_id、user id、object id、correlation id——各种“ID”不要再混

### 这一部分讲什么

rules 里到处都是 id，但其实不是一回事。
 这部分专门把“各种 ID”拆开。

### 会讲的核心概念

- client_id
- user identity
- subject identity
- object ID
- API key as client identifier
- correlation ID
- key identifier (kid)

### 在哪个环节会用到

当你看：

- OAuth client 配置
- BOLA / IDOR
- 网关审计
- 分布式日志
- JWT 校验

### 涉及哪些 rules

- API2-R10
- API1-R1
- API6-R1
- API10-R1
- API7-R4

------

## 第 9 部分：Token 安全——生命周期、格式、存储、撤销

### 这一部分讲什么

这一部分专门对应 token 相关 rules。

### 会讲的核心概念

- Token lifespan
- Token format
- Opaque token
- JWT / JWS / JWE
- Token storage
- Token rotation
- Revocation
- One-time use
- Short-lived token
- Token leakage

### 在哪个环节会用到

当你看：

- 前端是不是把 token 放 localStorage
- 后端是否安全保存 refresh token
- access token 是否太长
- token 是否出现在 URL / 日志
- token 是否可撤销

### 涉及哪些 rules

- API2-R2
- API2-R3
- API2-R5
- API6-R1
- API0-R2

------

## 第 10 部分：TLS、mTLS、证书、JKS、SSL bundle——传输层和证书层

### 这一部分讲什么

很多人把 token 和 TLS 混在一起。
 这部分专门讲“通信链路本身的安全”。

### 会讲的核心概念

- TLS
- HTTPS
- Certificate
- Certificate chain
- Pinning
- mTLS
- Client certificate
- Keystore
- Truststore
- JKS
- SSL bundle

### 在哪个环节会用到

当你看：

- RestTemplate / WebClient 配置
- 网关到后端的 mTLS
- APIM 到内部服务的连接
- 证书链校验
- 服务端调用外部 OAuth endpoint 的配置

### 涉及哪些 rules

- API0-R1
- API2-R7
- API0-R2
- API7-R4

------

## 第 11 部分：nonce、state、CSRF、replay attack——请求为什么“不是旧的、不是伪造的”

### 这一部分讲什么

这是安全参数和攻击防护部分。

### 会讲的核心概念

- Replay attack
- Nonce
- State
- CSRF
- One-time use
- Freshness
- Request/response binding

### 在哪个环节会用到

当你看：

- OIDC 登录回调
- ID token 校验
- 前端发起认证请求
- cookie-based 流程
- 敏感请求保护

### 涉及哪些 rules

- API6-R1
- API7-R5
- API2-R6

------

## 第 12 部分：API Gateway / Apigee / Axway 到底管什么，不管什么

### 这一部分讲什么

你现在做 assessment，网关是核心之一。
 但很多人会误以为“有网关就全合规”。

### 会讲的核心概念

- API Gateway
- API proxy
- Exposition layer
- Routing
- Header forwarding / stripping
- Request validation
- Response filtering
- Policy enforcement
- Centralized controls

### 在哪个环节会用到

当你判断：

- 流量是否必须过 gateway
- gateway 能做哪些粗粒度控制
- 哪些事情必须交给 business service
- Apigee 原则能覆盖哪些 rules

### 涉及哪些 rules

- API0-R3
- API1-R1
- API4-R1
- API7-R1
- API7-R2
- API7-R3
- API8-R1
- API10-R1

------

## 第 13 部分：输入校验、输出控制、CORS、rate limit、error handling

### 这一部分讲什么

这是“网关和后端最常落地的控制”。

### 会讲的核心概念

- Schema validation
- Input sanitization
- Output filtering
- Excessive data exposure
- Mass assignment
- CORS
- Rate limiting
- Quotas
- Error sanitization
- File upload constraints

### 在哪个环节会用到

当你看：

- OpenAPI / Swagger
- Apigee policy
- Spring validation
- 请求参数处理
- 响应字段裁剪
- CORS 配置
- 错误响应

### 涉及哪些 rules

- API3-R1
- API3-R2
- API3-R4
- API4-R1
- API7-R1
- API7-R2
- API7-R3
- API8-R1

------

## 第 14 部分：越权与暴露——BOLA、IDOR、technical API、predictable path

### 这一部分讲什么

这是很多 API assessment 的“真正高风险区”。

### 会讲的核心概念

- BOLA
- IDOR
- Field-level authorization
- Object-level authorization
- Technical API
- Predictable Resource Location
- Internal vs external exposure
- Admin vs user endpoints

### 在哪个环节会用到

当你判断：

- 改个对象 ID 会不会越权
- 管理接口是不是暴露到了互联网
- 技术接口是不是走了外部入口
- 路径和对象 ID 是否太可猜

### 涉及哪些 rules

- API1-R1
- API3-R3
- API7-R4
- API3-R1
- API3-R4

------

## 第 15 部分：日志、监控、追踪、SIEM——出了问题能不能看到

### 这一部分讲什么

安全不只是“拦住”，还要“看得见、追得到”。

### 会讲的核心概念

- Logging
- Traceability
- Correlation ID
- SIEM
- Monitoring
- Alerting
- Sensitive data in logs
- Log retention
- Log integrity

### 在哪个环节会用到

当你判断：

- 攻击是否能追踪
- 越权尝试是否会记录
- token 是否被错误打入日志
- 日志是否集中化

### 涉及哪些 rules

- API10-R1
- API7-R6
- API7-R7

------

## 第 16 部分：安全测试与治理——SAST、DAST、pentest、inventory、ownership

### 这一部分讲什么

最后才讲治理和流程，因为它建立在前面技术概念之上。

### 会讲的核心概念

- SAST
- DAST
- Vulnerability testing
- Penetration testing
- API inventory
- API registry
- IT owner / Business owner
- Unused API retirement
- Major release testing
- Go-live checklist

### 在哪个环节会用到

当你判断：

- 项目是否有安全测试机制
- 是否满足上线前测试要求
- API 是否有人负责
- API 是否被盘点和分级
- 不用的 API 是否下线

### 涉及哪些 rules

- API7-R6
- API7-R7
- API8-R2
- API9-R1
- API9-R2

------

## 第 17 部分：最后把所有概念串成“怎么做项目 assessment”

### 这一部分讲什么

前面都学完后，最后把它们串成一个实战方法。

### 会讲的内容

- 拿到一个项目时先看什么
- 架构图怎么读
- 代码里哪些类最关键
- 配置里优先看哪些项
- Apigee / WebSSO / Spring Boot / Angular 分别看什么
- 规则怎么快速映射到证据
- 怎么写 compliant / partially compliant / non-compliant / unknown

### 涉及哪些 rules

全部 rules

------

# 推荐讲解顺序

我建议我们严格按这个顺序走，不要跳：

1. 第 0 部分：assessment 总框架
2. 第 1 部分：参与方与系统角色
3. 第 2 部分：认证与授权
4. 第 3 部分：session / SSO
5. 第 4–6 部分：OAuth/OIDC、flow、token
6. 第 7–9 部分：claims、ID、token 管理
7. 第 10–11 部分：TLS/mTLS 与 nonce/state
8. 第 12–14 部分：gateway、输入输出控制、越权暴露
9. 第 15–16 部分：日志、测试、治理
10. 第 17 部分：assessment 实战方法



# 第 0 部分：Assessment 总框架

这一部分先不堆术语。
 我们先把你脑子里的“任务模型”搭起来，不然后面学再多概念也会散。

------

## 一、你现在到底在做什么

你不是在做这些事：

- 不是单纯学理论
- 不是单纯看代码有没有 `OAuth`、`JWT`、`mTLS` 这些关键词
- 不是单纯写一句“项目用了 Apigee，所以符合”
- 不是单纯证明“系统能跑”

你真正要做的是：

> **判断一个项目，是否用合适的方式、在合适的层级、用足够的证据，满足某条 API security rule。**

这句话你先记住，因为后面所有分析都围绕它。

------

## 二、什么叫“规则合规”，不是“有这个技术就行”

这是 assessment 最容易犯的错误。

### 例子 1：项目用了 OAuth2

这不自动等于符合 API0-R2。

为什么？
 因为 rule 可能要求的是：

- 必须是正确的 OAuth/OIDC flow
- 凭证不能落到浏览器
- BFF 模式下要 server-side 管理
- token 使用方式要符合要求

所以：

- **“用了 OAuth2”** 只是一个信号
- **“符合 rule”** 需要看它怎么用、在哪一层用、有没有绕过

------

### 例子 2：项目有 Apigee

这不自动等于符合 API0-R3、API4-R1、API7-R1、API8-R1。

为什么？
 因为你还要看：

- 外部流量是否真的都经过 Apigee
- rate limit 是否真的配置了
- CORS 是否真的收紧了
- schema validation 是否真的启用了
- backend 是否还有旁路入口

所以：

- **“有 Apigee”** 不等于 **“规则已落地”**

------

### 例子 3：代码里有 token

这不自动等于符合 token rules。

你还要看：

- token 是哪一种
- 生命周期多长
- 存在哪
- 是否可撤销
- 是否会打进日志
- 是不是 Bearer
- 是不是只给对的 API 用

------

## 三、你做 assessment 时，脑子里要有 5 层

这个是今天最重要的一张图。

------

### 第 1 层：Rule（规则层）

**通俗定义**：公司或集团要求你满足的安全规则。
 **例子**：必须使用 OAuth2/OIDC、必须做 rate limiting、不能暴露 technical APIs。

这一层回答的是：

> **规则到底要求什么？**

------

### 第 2 层：Architecture（架构层）

**通俗定义**：系统从大结构上，哪些组件负责哪些安全职责。
 **例子**：

- WebSSO 负责登录和签发 token
- Apigee 负责 gateway 控制
- Spring Boot 后端负责细粒度授权
- Angular 只是前端展示，不保存敏感凭证

这一层回答的是：

> **这条 rule 应该由系统的哪一层来负责？**

------

### 第 3 层：Implementation（实现层）

**通俗定义**：代码、配置、网关策略、部署设置，真正怎么实现。
 **例子**：

- Java 里通过 keystore 生成 assertion 去换 token
- Apigee policy 里启用了 VerifyJWT
- Spring Security 里对 admin endpoint 加了限制
- CORS 只允许特定 origin

这一层回答的是：

> **项目具体是怎么做的？**

------

### 第 4 层：Evidence（证据层）

**通俗定义**：你拿什么证明“它真的做了”。
 **例子**：

- 代码片段
- yml/properties 配置
- Apigee policy 截图
- OpenAPI 文档
- 网关产品配置
- 日志
- 架构图
- 安全测试报告

这一层回答的是：

> **我凭什么下这个结论？**

------

### 第 5 层：Assessment Conclusion（结论层）

**通俗定义**：最后你怎么判。
 常见会有：

- Compliant
- Partially Compliant
- Non-Compliant
- Unknown / Evidence Missing

这一层回答的是：

> **基于现有证据，这条 rule 最后怎么判？**

------

## 四、你以后分析每条 rule，都按同一个动作走

以后你不用每次都慌。
 只要按这个 4 步走就行。

------

### 第一步：先问，这条 rule 在要求什么

不要急着看代码，先把 rule 说成人话。

比如：

- API0-R3 说的是：外部 API 必须过 gateway
- API2-R8 说的是：会话要安全管理
- API3-R1 说的是：只给必要数据

如果这一步没想清楚，后面都会跑偏。

------

### 第二步：再问，这条 rule 应该落在哪一层

这是最关键的一步。

因为不是每条 rule 都该去 Apigee 找答案。

#### 例如

**OIDC / OAuth2 flow**
 主要看：

- WebSSO / Authorization Server
- client 接入方式
- 后端/BFF 拿 token 的方式

**Rate limiting**
 主要看：

- Apigee / API Gateway

**Fine-grained authorization**
 主要看：

- 后端业务服务
- Spring Security / service logic
- 必要时再看 gateway 粗粒度控制

**SSO lifespan**
 主要看：

- WebSSO / IdP
- session policy

**Unused API**
 主要看：

- 治理流程 / inventory / API owner

所以你一定要学会一句话：

> **不是所有 rule 都在同一层找。**

------

### 第三步：去找证据

你要开始形成“看到一条 rule，就知道去哪找”的习惯。

#### 可能的证据来源

**代码里找**

- Java service / config / controller / security config
- Angular auth logic
- Token 获取逻辑
- Validation / authorization logic

**配置里找**

- application.yml / properties
- K8S env / secrets
- Spring SSL bundle
- CORS config
- timeout / TTL config

**平台上找**

- Apigee policy
- WebSSO client registration
- certificate / keystore setup
- API product / app binding

**文档里找**

- 架构图
- OpenAPI
- security guideline
- pentest report
- SAST/DAST report
- API inventory

------

### 第四步：最后才下结论

这一步不能太快。

#### 什么时候可以判 Compliant

- rule 要求清楚
- 系统负责层找对了
- 有明确证据
- 证据足以说明关键控制点已经落地

#### 什么时候只能判 Partial

- 做了一部分
- 或方向对，但缺关键细节
- 或一层做了，但另一层缺了
- 或有控制，但不够强

#### 什么时候判 Non-Compliant

- 明显没做
- 或与 rule 正面冲突
- 或存在明显绕过/暴露/高风险缺口

#### 什么时候判 Unknown

- 还没有足够证据
- 不能靠猜

这点在企业里很重要：

> **证据不够，不要硬判合规。**

------

## 五、你以后会反复遇到的一个核心问题：这件事到底是谁负责

这是 assessment 里最值钱的能力。

------

### 1）有些控制属于 WebSSO / IdP

比如：

- 登录
- OIDC flow
- ID token
- SSO lifespan
- MFA
- nonce / state
- refresh token policy

相关 rules 例如：

- API0-R2
- API2-R4
- API2-R6
- API2-R9
- API6-R1

------

### 2）有些控制属于 Apigee / Gateway

比如：

- gateway must be in exposition layer
- CORS
- rate limit
- header filtering
- error sanitization
- routing exposure
- request schema validation（有时）
- logging / monitoring（部分）

相关 rules 例如：

- API0-R3
- API4-R1
- API7-R1
- API7-R2
- API7-R3
- API7-R4
- API8-R1

------

### 3）有些控制属于后端业务服务

比如：

- fine-grained authorization
- object-level authorization
- field-level scoping
- need-to-know
- business validation
- secret data exposure control

相关 rules 例如：

- API1-R1
- API3-R1
- API3-R2
- API3-R4
- API5-R1
- API8-R1

------

### 4）有些控制属于前端 / BFF / Client

比如：

- browser 中 token 怎么存
- 是否走 BFF
- logout
- session handling
- redirect handling
- PKCE in SPA/native app

相关 rules 例如：

- API0-R2
- API2-R1
- API2-R5
- API2-R8

------

### 5）有些控制属于治理和流程

比如：

- API inventory
- owner registry
- unused API retirement
- pentest frequency
- vulnerability test frequency
- SAST before go-live

相关 rules 例如：

- API7-R6
- API7-R7
- API8-R2
- API9-R1
- API9-R2

------

## 六、为什么你总会觉得“好像都有关，但又不是一回事”

因为同一条 rule，往往要跨层验证。

### 例子：API0-R2

你可能需要同时看：

- WebSSO：是不是标准 OIDC/OAuth2
- 前端/BFF：凭证是否放对位置
- 后端：是不是 server-side 拿 token
- Apigee：是不是接收正确的 Bearer token

所以一条 rule 不是“找一个地方就完”。

------

### 例子：API1-R1

你可能需要同时看：

- Apigee：scope 这种粗粒度授权
- 后端：对象级、字段级、业务级授权
- token：里面有没有身份信息
- 请求参数：有没有越权风险

------

## 七、你以后判断一条 rule 时，建议固定写法

你后面无论是口头分析还是写 assessment，都按这个顺序最稳。

### 1. Rule intent

这条 rule 到底在要求什么。

### 2. Relevant control layer

这条 rule 主要应该在哪一层落地。

### 3. Observed implementation / evidence

项目里看到了什么实现和证据。

### 4. Gap analysis

还有什么缺口，哪些点证据不足。

### 5. Conclusion

最终结论：Compliant / Partial / Non-Compliant / Unknown

------

## 八、一个最简单的示范

我们拿 **API0-R3** 举例。

### Rule intent

所有 external-facing API 必须通过 API Provider / Gateway。

### Relevant control layer

主要看 Apigee / API Gateway / exposure layer。

### Evidence

- 架构图显示外部流量先到 gateway
- Apigee 上有 API proxy
- 没有发现 backend 直连暴露地址
- Swagger 用的是 gateway 域名

### Gap analysis

如果还没确认是否存在旁路入口，就不能完全放心。

### Conclusion

如果证据充分，可判 Compliant；
 如果只看到架构图但没看到实际暴露路径，可能只能判 Partial 或 Unknown。

------

## 九、这一部分最重要的 5 句话，你先背下来

### 第一句

**Rule 不是技术名词列表，而是安全要求。**

### 第二句

**有某个技术，不等于符合某条 rule。**

### 第三句

**先判断这条 rule 应该在哪一层落地，再去找证据。**

### 第四句

**代码、配置、平台策略、文档、测试报告，都是证据。**

### 第五句

**证据不够时，宁可判 Unknown / Partial，也不要硬判合规。**

------

## 十、一句话总结这节

你做 API security assessment 的核心工作，不是背术语，而是把每条 rule 映射成一条链：
 **规则要求什么 → 应该由哪一层负责 → 项目里怎么实现 → 我看到了什么证据 → 最后怎么判断。**



# 第 1 部分：系统里的角色和参与方

这一部分你先不要追求“特别专业”，你只要先做到：

1. 看到一个组件，知道它大概属于哪一类
2. 看到一条 rule，知道它主要是约束哪一类角色
3. 后面看代码和架构图时，不会再把前端、BFF、gateway、认证中心、业务服务混成一坨

------

## 一、先给你一张最小脑图

一个企业 API 系统，通常至少会有下面这些参与方：

- **User**：真正使用系统的人
- **Client**：发起请求的应用
- **OpenID Provider / WebSSO**：负责登录、发 token
- **API Gateway / API Provider**：统一入口，做网关控制
- **Business Service Provider**：真正处理业务逻辑的后端服务
- **BFF（有时有）**：前端专用的服务端中间层

你先把它想成一条链：

**User → Client →（有时先到 BFF）→ API Gateway → Business Service**
 如果需要登录和发 token，中间还会跟 **WebSSO / OpenID Provider** 打交道。

------

# 二、User：真正使用系统的人

## 1）是什么

**User** 就是最终使用系统的人。
 不是程序，不是浏览器，不是服务，而是“人”。

## 2）银行/企业里的例子

- 个人客户
- 企业客户
- 柜员
- 风控人员
- 运营人员
- 内部员工

## 3）为什么这个概念重要

因为很多 API 请求不是“应用自己随便调”，而是：

> **应用代表某个 user 去调 API**

比如：

- 客户在前端点“查看账户”
- Angular 前端发请求
- 但这个请求背后真正代表的是“这个客户本人”

所以后端最后要判断的不只是：

- 这个 app 是谁

还要判断：

- 这个 user 是谁
- 这个 user 能不能看这个数据

## 4）在 assessment 里什么时候会用到

你一看到以下问题，就要想到 user：

- 这个 token 是代表 user 还是代表 app？
- 这个接口是给客户用的还是给管理员用的？
- 这个接口是不是只允许用户看自己的对象？
- MFA 要不要上？

## 5）主要涉及哪些 rules

- **API1-R1**（fine-grained authorization）
- **API3-R1**（need-to-know）
- **API3-R4**（least privilege）
- **API2-R8**（session management）
- **API2-R9**（end-user authentication）

------

# 三、Client：发起请求的应用

## 1）是什么

**Client** 是发请求的“应用程序”或“软件组件”。
 它不一定是前端，也不一定是浏览器。

## 2）常见例子

- Angular 前端
- 手机 App
- BFF
- Spring Boot 后端服务
- 内部 batch job
- partner application

## 3）最容易混淆的点

### Client 不是 User

- **User**：张三
- **Client**：手机银行 App

### 一个请求里两者可以同时存在

比如：

- User 是张三
- Client 是 Angular 前端
- Angular 代表张三去调 API

所以 OAuth/OIDC 世界里，经常会同时管：

- client identity
- user identity

## 4）为什么这个概念重要

因为很多 rules 都不是单纯看“用户登录了没”，而是在看：

- 这个 **client** 是什么类型
- 它能不能安全保存秘密
- 它适合什么 flow
- 它允许拿什么 scope
- 它是不是唯一标识

## 5）在 assessment 里什么时候会用到

你看到这些问题就要想到 client：

- 这是 public client 还是 confidential client？
- 它是否有自己的 `client_id`？
- 它是 browser-based 还是 native app？
- 它代表自己调 API，还是代表 user 调 API？

## 6）主要涉及哪些 rules

- **API2-R1**（grant flow）
- **API2-R7**（client authentication）
- **API2-R10**（client identification）
- **API0-R2**（authorization protocol）
- **API5-R1**（scope）

------

# 四、Public Client：不可信客户端

## 1）是什么

**Public Client** 指的是：
 **无法可靠地安全保存长期秘密的客户端。**

## 2）通俗理解

代码跑在用户设备上，或者跑在浏览器里，意味着：

- 代码可能被看到
- 配置可能被读到
- secret 不适合放在这里

## 3）常见例子

- 浏览器里的 SPA（例如 Angular）
- 一般的手机端前端代码
- 某些桌面客户端

## 4）为什么重要

因为很多认证方案会区分：

- public client 能不能有 `client_secret`
- public client 能不能保存 refresh token
- public client 应该用什么 flow
- public client 是否必须配 PKCE

## 5）你在项目里怎么认它

如果一个组件是：

- 运行在浏览器里
- 代码会发到用户设备上
- 用户可以抓包、看前端 JS

那它通常就应当按 **public client** 来看。

## 6）涉及哪些 rules

- **API0-R2**
- **API2-R1**
- **API2-R2**
- **API2-R5**
- **API2-R6**

------

# 五、Confidential Client：可信客户端

## 1）是什么

**Confidential Client** 指的是：
 **能够安全保存秘密的服务端客户端。**

## 2）通俗理解

如果代码跑在服务端，通常更容易做到：

- 私钥只放服务端
- client secret 不暴露给用户
- token 获取流程在后端完成

## 3）常见例子

- Spring Boot BFF
- 后端服务
- 内部服务调用程序
- batch / daemon

## 4）为什么重要

因为很多企业规则默认：

- confidential client 可以走更强的客户端认证
- 可以使用服务端证书或 JWT auth
- 可以承担更敏感的 token 管理职责

## 5）你在项目里怎么认它

如果它：

- 跑在服务端
- 有自己的配置文件/secret
- 可以访问 JKS / keystore / Vault
- 不把敏感凭证暴露给浏览器

那通常可以看作 confidential client。

## 6）涉及哪些 rules

- **API0-R2**
- **API2-R1**
- **API2-R7**
- **API2-R5**

------

# 六、OpenID Provider / Authorization Server / WebSSO：登录和发 token 的地方

这几个概念经常一起出现，我先帮你分开再合起来。

------

## 1）OpenID Provider（OP）是什么

**通俗定义**：
 负责用户认证、签发 ID token / access token 的身份系统。

## 2）Authorization Server 是什么

**通俗定义**：
 负责按 OAuth2 规则发 token 的服务器。

## 3）WebSSO 是什么

**通俗定义**：
 你们公司统一登录平台的业务/企业叫法。
 它背后在技术上很可能就是一个承担 OP / Authorization Server 职责的身份平台。

## 4）你现在可以怎么先理解

在你当前任务里，可以先粗略这样记：

> **WebSSO / OpenID Provider / Authorization Server**
>  都可以先理解成“负责登录和发 token 的身份中心”。

后面学细了再区分：

- OIDC 更偏认证身份
- OAuth2 更偏授权获取 token

## 5）为什么重要

因为很多 rules 其实不是让 Apigee 决定，而是让这个身份系统决定：

- 用户怎么登录
- MFA 要不要上
- token 发什么
- token 活多久
- refresh token 给不给
- SSO session 多久失效
- redirect_uri 合不合法
- nonce / state 怎么处理

## 6）在 assessment 里什么时候会用到

你一看到下面这些问题，就优先想到 WebSSO / OP：

- 用的是什么 flow
- token 从哪发出来
- SSO 生命周期多长
- 是否支持 PKCE
- 是否有 MFA
- ID token / access token 是什么格式
- redirect_uri 怎么校验

## 7）涉及哪些 rules

- **API0-R2**
- **API2-R1**
- **API2-R2**
- **API2-R4**
- **API2-R5**
- **API2-R6**
- **API2-R9**
- **API6-R1**
- **API7-R5**

------

# 七、API Provider / API Gateway：对外 API 的统一入口

这两个词在你现在任务里几乎可以先放在一起理解。

------

## 1）API Gateway 是什么

**通俗定义**：
 所有外部 API 请求进入系统前，先经过的统一入口。

## 2）API Provider 是什么

在你这套 rules 里，**API Provider** 往往就是承担 API 管理和暴露职责的平台。
 在实践中，通常就是：

- Apigee
- Axway API Manager
- Kong
- Nginx API gateway（更轻）
- 其他 API management 平台

## 3）你现在可以怎么先理解

> **API Gateway / API Provider**
>  先统一理解成：**对外 API 的前门**。

## 4）为什么重要

因为很多 rule 默认它负责做这些事：

- 所有外部流量先过它
- token 基础校验
- scope 粗粒度控制
- CORS
- rate limiting
- header 处理
- error sanitization
- routing
- monitoring / logging（部分）

## 5）它不负责什么

这是 assessment 最关键的一点之一。

Gateway **不等于**：

- 业务细粒度授权全在它做
- 用户会话全在它做
- 所有安全规则都只靠它实现

很多时候 gateway 只能做：

- **第一道门**
- **粗粒度控制**
- **统一入口治理**

真正细粒度业务控制还得看后端。

## 6）在 assessment 里什么时候会用到

你看到这些问题，就要想到 gateway：

- 外部流量是否都经过它
- 有没有旁路入口
- 有没有 rate limit
- CORS 是否收紧
- header 是否处理了
- Swagger/OpenAPI 是否走 gateway
- schema validation 是否启用

## 7）涉及哪些 rules

- **API0-R3**
- **API1-R1**（粗粒度部分）
- **API4-R1**
- **API7-R1**
- **API7-R2**
- **API7-R3**
- **API7-R4**
- **API8-R1**
- **API10-R1**

------

# 八、Business Service Provider：真正处理业务逻辑的后端

## 1）是什么

**通俗定义**：
 真正处理客户数据、账户数据、贷款数据、风控逻辑的后端服务。

## 2）常见例子

- customer-service
- account-service
- loan-service
- payment-service
- profile-service

一般都是：

- Spring Boot
- Java
- 微服务
- 后端 REST API

## 3）为什么重要

因为 API Gateway 不应该承担所有细粒度授权。
 很多企业规则都会强调：

> gateway 做粗粒度
>  backend 做细粒度

比如：

- gateway 能检查你有没有 `read_accounts` 这个 scope
- 但 backend 要检查你是不是只能看“你自己的 accountId”

## 4）在 assessment 里什么时候会用到

你看到下面这些，就要想到 business service：

- object-level authorization
- field-level authorization
- input semantic validation
- excessive data exposure
- secret data not exposed
- business rules
- admin vs user data restrictions

## 5）涉及哪些 rules

- **API1-R1**
- **API3-R1**
- **API3-R2**
- **API3-R4**
- **API5-R1**
- **API8-R1**

------

# 九、BFF（Backend for Frontend）：前端专用后端

这个概念你后面做 assessment 很常会遇到。

------

## 1）是什么

**通俗定义**：
 专门服务某个前端的后端中间层。

## 2）为什么会有它

因为浏览器前端经常不适合：

- 保存敏感凭证
- 直接调多个后端
- 管复杂会话
- 做 server-side aggregation

于是就会加一个 BFF：

- 浏览器只跟 BFF 说话
- BFF 再去调下游 API

## 3）典型链路

**User → Browser Angular → BFF → Apigee / Backend API**

## 4）为什么它在 rules 里重要

因为 **API0-R2** 明确提到了：

如果 browser-based application 需要 server-side session，可以允许 BFF 模式，但要满足：

- 浏览器和 BFF 之间用 cookie/session
- BFF 去请求 API 仍然要遵循 OIDC/OAuth2
- 凭证必须只在服务端管理，不能过浏览器

## 5）在项目里怎么认它

你看到这些现象时，通常说明有 BFF：

- 前端只请求 `/api/...` 相对路径
- Spring Boot 同时接前端和下游 API
- 浏览器不直接拿业务 API token
- cookie/session 在前端和中间层之间

## 6）涉及哪些 rules

- **API0-R2**
- **API2-R1**
- **API2-R5**
- **API2-R8**

------

# 十、再把这些角色串成几条典型调用链

你现在不要只记定义，要开始看“它们怎么串起来”。

------

## 场景 1：浏览器前端直接调 gateway

### 链路

User → Angular SPA → WebSSO 登录 → 拿 token → 调 Apigee → Apigee 转发给后端

### 特点

- 前端知道 gateway 地址
- token 可能在前端内存里
- gateway 是统一入口
- backend 做细粒度授权

### 重点 rules

- API0-R2
- API0-R3
- API2-R1
- API2-R5
- API7-R1
- API1-R1

------

## 场景 2：浏览器前端走 BFF

### 链路

User → Browser → BFF → WebSSO / token / downstream API → backend

### 特点

- 浏览器不直接碰敏感凭证
- BFF 是 confidential client
- 前端和 BFF 常用 cookie/session
- BFF 再去调用 Apigee 或 backend

### 重点 rules

- API0-R2
- API2-R1
- API2-R5
- API2-R8

------

## 场景 3：服务到服务调用

### 链路

Spring Service A → 拿 token / 用 mTLS → 调 Apigee 或 Service B

### 特点

- 没有 user 参与
- 常是 confidential client
- 常用 client_id、certificate、JWT assertion
- 常见于内部系统集成

### 重点 rules

- API2-R1
- API2-R7
- API0-R1
- API2-R2
- API2-R3

------

# 十一、你后面看架构图时，先做这 5 个判断

这部分非常实用，你可以直接拿去用。

------

## 判断 1：谁是 User

有没有“最终人类使用者”参与？
 如果有，是客户、员工、管理员，还是 partner user？

------

## 判断 2：谁是 Client

发请求的是：

- 浏览器前端
- 手机 App
- BFF
- 后端服务
- batch

------

## 判断 3：谁负责登录和发 token

是 WebSSO / OpenID Provider 吗？
 还是项目自己乱发 token？

------

## 判断 4：谁是统一入口

是 Apigee / gateway 吗？
 还是外部能直接打 backend？

------

## 判断 5：谁负责最终业务权限判断

是后端业务服务吗？
 还是项目把授权全想当然交给 gateway 了？

------

# 十二、一个最常见的错位，你现在先避免

很多新人会这样想：

- WebSSO = 所有认证授权都搞定了
- Apigee = 所有 API 安全都搞定了
- Spring Security = 所有业务安全都搞定了

这三种想法都不对。

你要开始建立的是“分工”：

### WebSSO / OP

负责：

- 登录
- 身份
- token issuance
- SSO/session policy（部分）

### Apigee / Gateway

负责：

- 统一入口
- 基础校验
- 粗粒度控制
- 暴露层治理

### Backend business service

负责：

- 真正业务数据
- 对象级/字段级授权
- 输入语义校验
- 最终数据暴露边界

------

# 十三、这一章最重要的 6 句话

## 第一句

**User 是人，Client 是应用。**

## 第二句

**Public Client 不适合保秘密，Confidential Client 更适合持有敏感凭证。**

## 第三句

**WebSSO / OpenID Provider 负责登录和发 token，不等于它负责全部授权。**

## 第四句

**API Gateway 是前门，不等于它替 backend 做完所有业务授权。**

## 第五句

**Business Service 才是最终处理业务数据和细粒度权限的地方。**

## 第六句

**BFF 是浏览器前端和后端 API 之间的一层服务端缓冲层。**

------

# 十四、一句话总结这章

你后面做 project assessment 时，第一步永远是先分清：
 **谁是用户、谁是客户端、谁负责登录发 token、谁是统一 API 入口、谁负责真正业务授权。**



# 第 2 部分：认证和授权，到底有什么区别

这一章是你后面看 rules、看 token、看代码时最重要的基础之一。
 因为如果这里混了，后面这些词你都会混：

- token
- scope
- role
- session
- fine-grained authorization
- least privilege
- need-to-know

所以我们这一章先把地基打牢。

------

## 一、先用一句最简单的话分开它们

### Authentication（认证）

> **证明你是谁。**

### Authorization（授权）

> **决定你能做什么。**

你现在先不要追求特别严谨，先把这两个感觉建立起来。

------

## 二、一个最贴近银行项目的例子

假设一个客户登录手机银行，想看账户信息。

### 第一步：认证

系统先确认：

- 这个人是不是张三
- 登录密码/短信验证码/生物识别是不是正确

这一步叫 **认证**。

### 第二步：授权

系统再确认：

- 张三能不能看这个账户
- 这个账户是不是张三自己的
- 张三是不是只能看余额，不能看某些内部字段

这一步叫 **授权**。

------

## 三、为什么这两个一定要分开

因为在项目里，**“登录成功”不等于“什么都能做”**。

很多真实问题是这样的：

- 用户确实登录了，认证没问题
- 但后端没有检查对象权限
- 用户把 URL 里的 `accountId` 改一下，就看到别人的账户

这时：

- 认证是对的
- 授权是错的

所以你以后看到安全问题，先别急着说“是不是没登录”，很多时候真正的问题是：

> **授权没做好。**

------

# 四、Authentication（认证）到底是什么

------

## 1）通俗定义

认证就是：

> 系统确认这次请求背后是谁。

它要回答的问题是：

- 这个 user 是谁？
- 这个 client 是谁？
- 这个请求是不是来自合法身份？

------

## 2）认证可以发生在两种对象身上

这个地方很重要，很多新人会漏掉一半。

### A. 对用户做认证

比如：

- 用户名密码
- MFA
- 生物识别
- WebSSO 登录

这时回答的是：

> 这个人是谁？

------

### B. 对客户端做认证

比如：

- client_id + secret
- JWT client authentication
- mTLS client certificate

这时回答的是：

> 这个应用是谁？

------

## 3）企业项目里，常见的认证对象组合

### 场景 1：只有客户端，没有用户

比如 batch job 调 API。
 系统关心的是：

- 哪个应用在调

这类场景常见于 service-to-service。

------

### 场景 2：既有客户端，也有用户

比如浏览器前端代表张三调 API。
 系统需要同时知道：

- 是哪个 app 在调
- 是代表哪个 user 在调

这类场景在你现在的 assessment 里特别常见。

------

## 4）认证常见的结果是什么

认证做完后，系统通常会得到这些信息：

- user identity（用户是谁）
- client identity（客户端是谁）
- authentication strength（认证强度够不够）
- token（后面要讲）
- session（后面要讲）

------

# 五、Authorization（授权）到底是什么

------

## 1）通俗定义

授权就是：

> 在已经知道“你是谁”之后，决定“你能做什么”。

它要回答的问题是：

- 你能不能调这个 API？
- 你能不能访问这个对象？
- 你能不能看这个字段？
- 你能不能做这个动作？

------

## 2）授权不是一个动作，而是一层一层的

这点特别重要。

很多人以为授权就是“有个角色判断一下就完了”，其实企业 API 里经常分成几层：

### 第一层：能不能进这个 API

比如你有没有 `read_accounts` scope。

### 第二层：能不能访问这个对象

比如这个 `accountId` 是不是你的。

### 第三层：能不能看这个字段

比如你可以看账户余额，但不能看某些内部风控字段。

### 第四层：能不能做这个动作

比如你能查询，但不能转账；或者你能编辑草稿，但不能审批。

------

# 六、认证和授权的区别，再用 4 个例子打牢

------

## 例子 1：登录系统

用户输入密码登录成功。

### 这是认证还是授权？

**认证。**

因为系统只是在确认这个人是谁。

------

## 例子 2：登录后访问管理员页面

用户已经登录了，但系统还要检查他是不是 admin。

### 这是认证还是授权？

**授权。**

因为系统已经知道你是谁了，现在在判断你能不能进 admin 页面。

------

## 例子 3：用户看自己的订单

用户已登录，但系统要确认订单是不是他的。

### 这是认证还是授权？

**授权。**

------

## 例子 4：后端服务用证书去换 token

服务端程序拿自己的证书或私钥去证明自己身份。

### 这是认证还是授权？

**认证。**

是客户端认证，不是用户认证。

------

# 七、和授权相关的几个核心概念

接下来这几个词，后面 rules 里会反复出现。

------

## 1）Permission（权限）

### 通俗定义

某个主体被允许做的一件事。

### 例子

- 允许读取客户资料
- 允许发起转账
- 允许查看贷款申请
- 允许访问 admin endpoint

### 在 assessment 里什么时候会用到

当你看：

- 接口权限设计
- Spring Security 注解
- role/scope 配置

------

## 2）Role（角色）

### 通俗定义

一类身份分组，通常代表职责。

### 例子

- customer
- admin
- operator
- advisor
- manager

### 为什么重要

角色经常用来做授权：

- admin 可以做什么
- customer 只能做什么
- internal operator 能做什么

### 但要注意

角色不是唯一授权方式。
 复杂系统里，光靠 role 往往不够。

------

## 3）Scope

### 通俗定义

一种比较粗粒度的“可访问能力范围”。

### 例子

- `read_accounts`
- `write_profile`
- `transfer_funds`

### 它和 role 的区别

这个你一定要先有个直觉：

- **Role** 更像“你是什么身份”
- **Scope** 更像“这次 token 被允许访问哪些能力”

### 为什么 scope 在 API 里特别常见

因为 gateway 很适合先按 scope 做第一层控制。

比如：

- 你这个 token 没有 `read_accounts`
- 那 gateway 就先不让你进账户 API

### 涉及哪些 rules

- API1-R1
- API5-R1
- API3-R4
- API2-R1

------

## 4）Need-to-know

### 通俗定义

只给你工作上真正需要的数据。

### 例子

客服查看客户信息时，只应看到联络信息，不一定应该看到内部风险评分。

### 它在项目里长什么样

往往体现为：

- 响应字段裁剪
- 不返回多余字段
- 不把整条对象全部返回给前端

### 涉及哪些 rules

- API3-R1
- API3-R2

------

## 5）Least privilege

### 通俗定义

只给完成任务所需的最小权限。

### 例子

一个只读报表系统，不应该拿到可写、可审批、可管理的权限。

### 它和 need-to-know 的区别

这两个很像，但你先这样记：

- **Need-to-know**：给多少数据
- **Least privilege**：给多大权限

### 涉及哪些 rules

- API3-R4
- API5-R1
- API1-R1

------

# 八、粗粒度授权和细粒度授权

这个是你做 assessment 时会特别常用的分法。

------

## 1）Coarse-grained authorization（粗粒度授权）

### 通俗定义

先判断你大方向上能不能进这个 API/服务。

### 例子

- 你有没有 `read_accounts` scope
- 你是不是这个 API 的 authorized client
- 你是不是 customer portal app

### 谁常做这件事

通常是：

- API Gateway / API Provider
- token-based scope check

### 为什么企业喜欢先在 gateway 做这一层

因为它统一、简单、适合大门口拦截。

------

## 2）Fine-grained authorization（细粒度授权）

### 通俗定义

你进来了以后，再判断你能不能访问这个具体对象、这个字段、这个动作。

### 例子

- 你能看账户 API，但只能看你自己的账户
- 你能看客户列表，但不能看隐藏字段
- 你能编辑草稿，但不能审批

### 谁常做这件事

通常是：

- Business service
- Spring Security + service logic + DB constraints

### 为什么不能只靠 gateway 做

因为 gateway 往往不知道：

- 这个 `accountId` 到底属于谁
- 这个字段是不是敏感字段
- 这个动作是不是某条业务规则下才允许

### 涉及哪些 rules

- API1-R1 是最核心的这一条
- API3-R1
- API3-R4
- API5-R1

------

# 九、RBAC 和 ABAC

这两个词你后面会经常见到。

------

## 1）RBAC（Role-Based Access Control）

### 通俗定义

按角色授权。

### 例子

- admin 可以访问管理接口
- customer 只能访问自己的基础功能
- operator 可以查看工单

### 优点

简单、直观、容易治理。

### 局限

复杂业务里，角色往往不够细。

比如：

- 两个都是 operator
- 但一个只能看西班牙区数据，一个只能看法国区数据

这时就不够了。

------

## 2）ABAC（Attribute-Based Access Control）

### 通俗定义

按属性授权。

### 例子

系统判断时不只看 role，还看：

- 国家
- 组织
- 客户归属
- 数据分类
- 渠道
- 时间
- 风险等级

### 银行场景例子

“只有属于 Madrid branch 且当前处理的是自己管辖客户的 advisor，才能访问该客户的完整资料。”

### 为什么企业喜欢 ABAC

因为它更贴近真实复杂业务。

------

## 3）RBAC 和 ABAC 的关系

你先这样理解就够了：

- **RBAC**：按“你是什么角色”
- **ABAC**：按“你具备什么属性、当前处于什么上下文”

很多系统其实是两者混用。

------

# 十、你以后怎么从代码里看“认证”和“授权”

这部分很实用。

------

## 1）看到这些，多半是认证相关

你以后在代码/配置里看到这些，先想到认证：

- login
- oauth
- oidc
- token endpoint
- WebSSO
- authentication manager
- mTLS
- client certificate
- JWT validation（有时候既相关认证也相关鉴权）
- session creation
- redirect_uri
- PKCE

------

## 2）看到这些，多半是授权相关

你以后在代码/配置里看到这些，先想到授权：

- role
- scope
- permission
- `@PreAuthorize`
- access denied
- account ownership check
- object-level validation
- field filtering
- admin vs user endpoint
- response masking

------

# 十一、你以后怎么从规则里看“认证”和“授权”

我帮你把常见映射先建立起来。

------

## 更偏认证的 rules

这些 rule 更偏“你是谁、怎么证明、怎么登录、token 怎么发”。

- API0-R2
- API2-R1
- API2-R2（部分）
- API2-R4
- API2-R5（部分）
- API2-R6
- API2-R7
- API2-R8（会话）
- API2-R9
- API6-R1
- API7-R5

------

## 更偏授权的 rules

这些 rule 更偏“你能做什么、能看什么”。

- API1-R1
- API3-R1
- API3-R2
- API3-R3
- API3-R4
- API5-R1

------

## 既有认证也有授权味道的 rules

有些 rules 两边都沾，比如：

- API0-R2
- API2-R5
- API6-R1

所以你不要期待每条 rule 都能绝对分到一边。

------

# 十二、一个你以后必须反复问自己的问题

每次你看到一个项目实现时，都先问：

## 问题 1：这是在解决“你是谁”吗？

如果是，就是认证向。

## 问题 2：还是在解决“你能做什么”？

如果是，就是授权向。

## 问题 3：还是两者都有？

那就分层看，不要混成一句话。

------

# 十三、一个完整的小例子，把这一章串起来

假设一个客户用 Angular 前端查看账户。

### 第一步：用户登录 WebSSO

系统确认这个人是张三。
 这是 **认证**。

### 第二步：前端拿到 token，调用 API Gateway

Gateway 看这个 token 有没有访问账户服务的 scope。
 这是 **粗粒度授权**。

### 第三步：请求到了 Spring Boot 后端

后端检查 `accountId` 是不是张三自己的账户。
 这是 **细粒度授权**。

### 第四步：后端构造响应

后端只返回页面需要的字段，不返回内部敏感字段。
 这体现的是 **need-to-know**。

### 第五步：系统没有给这个前端多余的高权限 scope

这体现的是 **least privilege**。

你看，一次请求里：

- 有认证
- 有授权
- 有粗粒度
- 有细粒度
- 有数据最小暴露
- 有权限最小化

这就是企业项目为什么看起来复杂。

------

# 十四、这一章最重要的 7 句话

## 第一句

**认证是“你是谁”，授权是“你能做什么”。**

## 第二句

**登录成功，不等于什么都能做。**

## 第三句

**很多 API 安全问题不是认证失败，而是授权失控。**

## 第四句

**Scope 常用于粗粒度授权，backend 负责细粒度授权。**

## 第五句

**Need-to-know 更偏“给多少数据”，least privilege 更偏“给多大权限”。**

## 第六句

**RBAC 按角色授权，ABAC 按属性和上下文授权。**

## 第七句

**Gateway 常做第一层授权，business service 做最终业务授权。**

------

# 十五、一句话总结这章

你以后做 assessment 时，看到任何安全实现，先问自己：
 **它是在证明“请求方是谁”，还是在限制“请求方能做什么”，还是两者都涉及。**



# 第 3 部分：会话、登录态、SSO，到底是什么

这一章你先不用急着记所有术语，先抓住一句话：

> **登录只是开始，会话管理决定这次登录接下来是不是安全。**

------

## 一、先说“登录态”是什么

### 1）通俗理解

当用户登录成功后，系统通常不会要求用户每点一次按钮都重新输密码。
 系统会在一段时间内“记住你是谁”。

这段被系统“记住”的状态，就是你可以先理解成的：

- 登录态
- 会话状态
- session state

### 2）银行场景例子

你登录网上银行后：

- 点“账户列表”
- 点“最近交易”
- 点“贷款信息”

这几次操作之间，你没有重新登录，说明系统一直在维持你的登录态。

------

## 二、Session（会话）是什么

### 1）定义

**Session（会话）** 就是系统在一段时间内维持“这个请求属于某个已登录主体”的状态机制。

### 2）通俗定义

你可以把 session 理解成：

> **系统给你开的一个临时“已登录窗口期”**

只要这个窗口期还有效，你接下来的请求就会被当成“同一个已登录用户/客户端”的后续操作。

### 3）它为什么重要

因为后面所有问题都围绕它展开：

- 多久失效
- 空闲多久后失效
- 退出后是否真的失效
- 多个设备能不能同时在线
- 改密码后旧会话是否还活着

------

## 三、Session 不一定只有一种实现方式

这是很容易混的地方。

很多新人以为 session 就等于“后端内存里存一份”。
 实际上，企业项目里有多种方式。

------

### 1）经典服务端 Session

#### 直观理解

后端服务器保存一份会话记录，浏览器只带一个 session id。

#### 场景例子

浏览器带着一个 cookie，里面只有一个会话标识，真正会话数据存在服务端。

#### 特点

- 服务端掌控更强
- logout / invalidation 通常更直接
- 适合传统 Web 应用和 BFF 模式

------

### 2）基于 Token 的“无状态会话感”

#### 直观理解

服务端不一定保存完整会话，而是靠 access token 识别请求方。

#### 场景例子

SPA 前端拿着 access token 调 API，后端每次都验 token。

#### 特点

- 更适合 API / 微服务
- 更容易扩展
- 但 token 生命周期、撤销、刷新就变得更关键

------

### 3）BFF + Cookie + Token 的混合模式

这在企业里很常见。

#### 长相

- 浏览器和 BFF 之间：cookie / session
- BFF 和 API / gateway 之间：token

#### 为什么常见

因为浏览器不适合保存太多敏感认证材料，所以把 token 留在服务端，让浏览器只持有 cookie/session。

#### 这和哪个 rule 强相关

- **API0-R2**
- **API2-R8**

------

## 四、Cookie 是什么，和 Session 什么关系

### 1）Cookie 是什么

**Cookie** 是浏览器存的一小段数据。

### 2）它最常见的安全用途

浏览器把 cookie 带给服务端，让服务端知道：

- 这是哪个已登录会话

### 3）Cookie 不等于 Session

这是很重要的一点。

- **Cookie**：浏览器里存的一小段东西
- **Session**：系统维持登录状态的整体机制

你可以这样记：

> Cookie 常常只是“会话的门票”，Session 才是“会话本身”

### 4）银行/企业项目里的典型情况

- 浏览器只保存 session cookie
- 真正的登录态在 WebSSO 或 BFF 服务端维护

------

## 五、SSO 是什么

### 1）定义

**SSO（Single Sign-On）** 是“单点登录”。

### 2）通俗理解

用户登录一次之后，在多个相关系统里不用重复登录。

### 3）银行场景例子

你登录公司门户后，再进某个内部操作台、API 文档平台、报表平台，不需要再次输入密码。

### 4）为什么企业喜欢 SSO

因为它能带来：

- 统一登录体验
- 统一身份策略
- 集中 MFA
- 集中 token / session 管理

### 5）为什么安全规则会特别管它

因为一旦 SSO 会话太长、退出不干净、被盗用，风险会被放大成：

> 一次身份失陷，多个系统都能进

这就是 **API2-R4** 很关注 SSO lifespan 的原因。

------

## 六、SSO 和普通 Session 的关系

你先这样记：

### 普通 Session

更像是：

- 某个具体应用里的登录态

### SSO Session

更像是：

- 身份中心（WebSSO / IdP）那边的统一登录态

它们可能同时存在。

------

### 一个很典型的企业调用链

1. 用户先在 WebSSO 登录，形成 **SSO session**
2. 某个前端/BFF/API 系统借这个 SSO 状态获得 token 或本地 session
3. 用户在应用里继续操作，形成应用自己的 session/token 使用状态

所以 assessment 时经常要分开问：

- WebSSO 的 SSO session 多久失效
- 应用自己的 session / access token 多久失效

------

## 七、Idle timeout 和 Absolute timeout

这两个词是会话管理的核心。

------

### 1）Idle timeout（空闲超时）

#### 定义

如果用户一段时间没操作，会话就失效。

#### 通俗理解

你什么都不点，系统等一会儿就把你“自动登出”。

#### 银行场景例子

客户打开账户页面后离开工位，20 分钟不操作，再回来就得重新登录。

#### 为什么重要

防止用户离开设备后，别人直接接着操作。

------

### 2）Absolute timeout（绝对超时）

#### 定义

无论用户是否一直在操作，只要到达最大时长，会话就必须失效。

#### 通俗理解

就算你一直在不停点击，到了 4 小时也得重新登录。

#### 为什么重要

防止一个会话无限续命、长期有效。

------

### 3）为什么两个都要有

如果只有 idle timeout：

- 用户只要偶尔点一下，就能无限维持会话

如果只有 absolute timeout：

- 用户离开电脑 30 分钟，别人还能继续操作，风险太高

所以企业里经常两者都要。

------

## 八、Logout（退出）到底意味着什么

### 1）很多人误以为 logout 很简单

以为：

- 前端跳回登录页
- 就算退出了

这不对。

### 2）真正安全的 logout 至少要回答 3 个问题

#### A. 前端状态清掉了吗

比如：

- 本地 token
- 本地 session 标记
- 内存里的认证状态

#### B. 服务端 session 失效了吗

如果服务端还记得你是登录状态，那只是“页面假退出”。

#### C. 身份中心 / OpenID Provider 那边的 token 或登录态失效了吗

如果 WebSSO 还保留有效状态，用户可能一刷新又自动回来了。

------

### 3）为什么这和 rules 强相关

这正是 **API2-R8 Session management** 的重点之一：

- 用户可以随时关闭当前 session
- logout 后服务端 session 必须失效
- access token 必须删除并撤销

------

## 九、Session invalidation（会话失效）是什么

### 1）定义

让一个原本有效的 session 立刻失效。

### 2）通俗理解

系统把“这张已登录门票”作废了。

### 3）常见触发场景

- 用户主动 logout
- 长时间空闲
- 超过绝对超时
- 密码修改
- 安全风险触发
- 管理员强制下线

### 4）为什么重要

如果会话不被真正 invalidated，就会出现：

- 假退出
- 旧浏览器标签页还能继续用
- 旧设备还能继续访问

------

## 十、Concurrent sessions（并发会话）是什么

### 1）定义

同一个用户能同时有多少个活跃会话。

### 2）通俗理解

一个账号能不能：

- 同时在电脑 A、电脑 B、手机、平板都登录着

### 3）为什么企业会管这个

因为如果完全不限制：

- 账号被盗后不容易发现
- 风险长期存在
- 旧设备退出不干净

### 4）相关 rule

- **API2-R8**

------

## 十一、Password change 后为什么要终止其他会话

### 1）逻辑很简单

如果用户改密码，通常意味着：

- 怀疑账号泄露
- 或者至少身份状态发生了重大变化

### 2）如果旧会话不失效会怎样

攻击者即使不知道新密码，也可能继续用旧 session 或旧 token。

### 3）为什么这是 session 管理要求的一部分

因为这属于：

> “凭证变化后，旧登录态必须收敛”

这也是 **API2-R8** 明确要求的点。

------

## 十二、SSO lifespan 和 access token lifespan 为什么要一起看

这就是 **API2-R4** 和 **API2-R8** 会互相关联的原因。

------

### 1）Access token 生命周期

它决定：

- 这个 token 多久不能再调用 API

### 2）SSO session 生命周期

它决定：

- 用户在身份中心那边的登录状态还能维持多久

------

### 3）为什么这两个不能完全脱节

如果 access token 很短，但 SSO session 很长：

- 前端可能不断静默拿新 token
- 用户表面上“很久没重新登录”，实际一直续命

如果 SSO session 很短，但 access token 很长：

- 用户在身份中心已失效
- 但已有 token 可能还在继续使用

所以企业规则会要求：

> 它们之间要协调，不要一边很短，一边无限长。

------

## 十三、浏览器前端、BFF、后端，在会话上的分工

这一点非常重要，因为很多项目不是纯一种模式。

------

### 场景 A：浏览器直接拿 token 调 API

#### 特点

- 浏览器维持前端登录态
- token 在前端内存/某种存储中使用
- session 更偏 token-based

#### 风险关注

- token 存哪
- 刷新机制怎么做
- logout 是否清干净
- refresh token 是否违规出现在浏览器

#### 相关 rules

- API0-R2
- API2-R5
- API2-R8

------

### 场景 B：浏览器 + BFF

#### 特点

- 浏览器和 BFF 之间常用 cookie/session
- BFF 去拿 token 调 API
- 敏感凭证留在服务端

#### 这通常是企业更保守的做法

因为浏览器不适合长期持有敏感认证材料。

#### 相关 rules

- API0-R2
- API2-R5
- API2-R8

------

### 场景 C：纯服务到服务

#### 特点

- 通常没有“用户会话”
- 更多是 client authentication + token lifecycle

#### 这时 session 这个概念弱一些

但 token 生命周期仍然重要。

------

## 十四、你做 assessment 时，这一章到底怎么用

以后当你判断某个项目是否符合会话类 rules 时，固定问下面这些问题。

------

### 1）这个项目有没有“登录后持续有效”的状态

如果有，它是靠什么维持的：

- cookie + server session
- token
- BFF session
- SSO session

------

### 2）空闲多久会失效

有没有 idle timeout。

------

### 3）最长多久必须重新登录

有没有 absolute timeout。

------

### 4）logout 是否是真退出

看：

- 前端是否清状态
- 服务端 session 是否失效
- token 是否撤销
- WebSSO 状态是否处理

------

### 5）同一个用户是否能无限设备同时在线

有没有 concurrent session limit。

------

### 6）改密码后旧会话是否会失效

如果不会，这是明显风险点。

------

### 7）SSO session 和 access token 是否协调

看：

- token TTL
- SSO idle/max time
- silent refresh 逻辑

------

## 十五、这一章和哪些 rules 对应最强

------

### API2-R8 — Session management

这是本章最直接对应的 rule。

它要求你检查：

- inactivity re-authentication
- logout
- server-side invalidation
- absolute timeout
- concurrent sessions
- password change invalidation

------

### API2-R4 — SSO lifespan

它关注的是：

- SSO idle time
- SSO max lifespan
- 和 token lifespan 的协调

------

### API0-R2 — BFF / browser-based app

它会间接涉及：

- 浏览器 session 管理
- BFF cookie/session
- 服务端凭证托管

------

### API2-R5 — Token management

如果项目是 token-based session 感，那它和本章也高度相关。

------

## 十六、一个完整的小例子，把这一章串起来

假设一个用户登录了 Angular + BFF + Apigee + Spring Boot 的系统。

### 第一步：用户登录 WebSSO

WebSSO 建立了一个 SSO session。

### 第二步：浏览器和 BFF 之间建立会话

浏览器拿到 cookie，BFF 在服务端维持登录态。

### 第三步：BFF 去拿 access token 调 Apigee

BFF 用服务端凭证去获取 token。

### 第四步：用户一段时间不操作

如果 idle timeout 到了，BFF session 应该失效。

### 第五步：用户点击 logout

浏览器 cookie 要清掉，BFF session 要 invalidated，必要时 token 要撤销，WebSSO 那边也要处理退出。

### 第六步：用户改密码

其他设备上的会话也应该被终止。

你看，这一整串都不是“登录那一下”的问题，而是会话生命周期问题。

------

## 十七、这一章最重要的 7 句话

## 第一句

**登录成功只是开始，会话管理决定后续是否安全。**

## 第二句

**Session 是系统维持“你已登录”状态的一种机制。**

## 第三句

**Cookie 只是浏览器里的一小段数据，不等于整个 session。**

## 第四句

**SSO 是身份中心层面的统一登录状态，不等于某个应用自己的会话。**

## 第五句

**Idle timeout 防长时间不操作被别人接手，absolute timeout 防会话无限续命。**

## 第六句

**真正的 logout 必须让前端状态、服务端会话、必要时的 token/SSO 状态都失效。**

## 第七句

**很多 session 类规则不在 Apigee 看，而在 WebSSO、BFF、前端和后端会话逻辑里看。**

------

## 十八、一句话总结这章

你以后看到任何“登录后还能继续用一段时间”的机制，都要开始问：
 **这个登录态是怎么维持的、多久失效、怎么退出、退出后是不是真的失效。**



# 第 4 部分：OAuth 2.0 / OIDC 是什么，为什么它们总出现

这一章的目标不是让你背 RFC，而是让你建立一个非常清晰的直觉：

1. 为什么 API 项目老提 OAuth2 / OIDC
2. 它们分别解决什么问题
3. 它们和你前面学的“认证、授权、session、client、user”怎么连起来
4. 你做 assessment 时，为什么必须先把这两个看懂

------

## 一、先说最简单的：为什么不能直接“账号密码调 API”

你可以先从一个最朴素的问题开始。

### 如果没有 OAuth2 / OIDC，会怎么样

最原始的做法可能是：

- 前端拿用户名密码
- 直接去调后端 API
- 或者每次请求都带账号密码

这有几个大问题：

### 问题 1：密码暴露面太大

密码可能出现在：

- 前端代码
- 请求链路
- 日志
- 多个系统之间

### 问题 2：权限太粗

系统很难优雅地表达：

- 这个 app 能看什么
- 这个 token 只能做什么
- 这个访问能活多久

### 问题 3：无法细粒度治理

你很难优雅地做到：

- token 短时有效
- 撤销某个 app 的权限
- 限制 scope
- 区分 client 和 user

### 所以现代系统的思路是

不要让各种应用到处拿用户密码乱飞，而是：

> **让专门的身份/授权系统来发一种可控的临时凭证，再让应用拿这个凭证调 API。**

这个“可控的临时凭证”，就是后面要讲的 token。

------

# 二、OAuth 2.0 到底是什么

------

## 1）先给一个小白版定义

**OAuth 2.0** 可以先理解成：

> 一套标准流程，用来让客户端以一种受控的方式获得访问 API 的权限，而不是直接拿着用户密码去到处调用。

你先不要急着想特别复杂。
 先记住它想解决的核心问题：

- 怎么拿到访问 API 的许可
- 怎么让这个许可可控、可过期、可限制范围

------

## 2）它到底在管什么

OAuth2 更偏向管：

- **访问权限**
- **令牌签发**
- **客户端如何拿 token**
- **token 如何代表某种访问许可**

所以你可以先把它和“授权”关联起来。

### 粗略一句话

OAuth2 更像在回答：

> **“这个客户端怎么合法拿到一个访问 API 的通行证？”**

------

## 3）它不直接等于“用户身份系统”

这是很多人最容易混的点。

OAuth2 本身更偏：

- access delegation
- token issuance
- API access control

它不是专门为“告诉你用户到底是谁”设计的。

所以后来大家就在 OAuth2 上面又加了一层更适合身份认证的标准，这就是 OIDC。

------

# 三、OIDC（OpenID Connect）是什么

------

## 1）小白版定义

**OIDC** 可以先理解成：

> 在 OAuth 2.0 之上，加了一套标准化的“用户身份认证结果”机制。

换句话说：

- OAuth2 更偏“拿访问 API 的许可”
- OIDC 更偏“标准化告诉客户端：这个登录的人是谁”

------

## 2）它为什么会出现

因为光有 OAuth2，客户端虽然能拿到 access token，但很多时候还需要知道：

- 用户是谁
- 用户是不是刚刚登录的
- 用户用了什么认证方式
- 这次认证强度够不够

于是 OIDC 引入了：

- ID token
- userinfo
- 一套更标准的认证结果表达方式

------

## 3）粗略一句话

OIDC 更像在回答：

> **“这个登录结果里，用户身份是什么，以及客户端该如何标准化拿到这份身份信息。”**

------

# 四、OAuth2 和 OIDC 的关系，到底怎么记

这个是最关键的理解点之一。

------

## 1）最简单记法

### OAuth 2.0

更偏：

- 授权
- API 访问许可
- access token
- grant flow

### OIDC

更偏：

- 认证
- 用户身份
- ID token
- 登录结果标准化

------

## 2）一个不太严谨但很有用的记忆法

你可以先记成：

- **OAuth2**：让你拿到“能进门”的票
- **OIDC**：再告诉你“拿票的人是谁”

------

## 3）为什么 rules 里经常把它们写在一起

因为现代企业 API 项目里，用户访问 API 常常既需要：

- 知道这个用户是谁
- 也需要发一个 access token 去调 API

所以规则里就会写成：

- OpenID Connect / OAuth 2.0 must be used

也就是把“认证 + API 访问许可”这两件事一起纳入标准流程。

------

# 五、OAuth2 / OIDC 系统里到底有哪些关键角色

这一部分会把前面第 1 章的角色，和这一章连起来。

------

## 1）Resource Owner

### 小白理解

资源真正的主人。

### 最常见情况

通常就是 user。

### 银行场景

客户自己的账户数据，资源所有者就是客户本人。

------

## 2）Client

前面学过了。
 这里再补一句：

在 OAuth2 / OIDC 语境里，**Client** 是去申请 token、然后拿 token 调 API 的应用。

可以是：

- Angular 前端
- Mobile App
- BFF
- 后端服务

------

## 3）Authorization Server / OpenID Provider

前面也学过。
 在这里它负责：

- 登录用户
- 验证客户端
- 决定发不发 token
- 发 access token / ID token

------

## 4）Resource Server

### 小白理解

真正持有资源、接收 access token 的 API 服务。

### 银行场景

账户服务、客户服务、贷款服务。

很多时候：

- Apigee 是前门
- 后端 business service 是最终 resource server

有时 gateway 先做部分校验，backend 再做最终处理。

------

# 六、为什么 OAuth2 / OIDC 特别适合 API

因为它非常适合表达这几件事：

------

## 1）不同 client 有不同权限

比如：

- mobile app 有一组 scope
- partner app 有另一组 scope
- internal batch 又有另一组

------

## 2）访问凭证可以短时有效

比如 access token 只活 15 分钟，而不是拿用户密码永久乱用。

------

## 3）可以把用户身份和客户端身份拆开

系统可以同时知道：

- 这是哪个 client 在调
- 它代表哪个 user 在调

------

## 4）可以支持不同场景

比如：

- 浏览器前端
- native app
- backend service
- device flow

不同场景用不同 flow。

------

## 5）可以和 gateway、scope、token、审计整合

这就是为什么你现在所有 rules 基本都围着它们打转。

------

# 七、Grant Flow 是什么，先给你一个预告版

下一章我们会细讲 flow。
 这里你先建立一个直觉。

------

## 1）Grant Flow 是什么

**Grant Flow** 就是：

> Client 通过什么方式拿到 token。

### 例子

- Authorization Code + PKCE
- Client Credentials
- Device Code
- Token Exchange

------

## 2）为什么 rules 很关心 flow

因为“有没有 token”不是重点，**怎么拿到 token** 才是重点之一。

举例：

- 浏览器前端不适合随便用某些 flow
- public client 和 confidential client 适合的 flow 不一样
- 用户参与和服务到服务调用适合的 flow 也不一样

------

# 八、Token 在 OAuth2 / OIDC 里扮演什么角色

这一章不展开所有 token 细节，先给你一个总感觉。

------

## 1）Access Token

用于调 API 的通行证。

### OAuth2 里最核心的东西

客户端拿到它之后，去请求 resource server / API。

------

## 2）ID Token

用于告诉客户端“用户身份结果是什么”。

### OIDC 才特别强调它

这也是 OIDC 比 OAuth2 多出来的一层。

------

## 3）Refresh Token

用于换新的 access token。

### 为什么重要

因为 access token 通常短时有效，但用户不想频繁重新登录。

------

## 4）为什么 token 比密码更适合 API

因为 token 可以做到：

- 更短命
- 更细权限
- 可撤销
- 不必把密码暴露给所有系统

------

# 九、为什么浏览器、BFF、后端服务，使用 OAuth2/OIDC 的方式会不一样

这个非常重要，因为 assessment 时你要判断“用法对不对”。

------

## 1）浏览器前端

特点：

- 不可信
- 代码会到用户设备
- 不适合长期持有秘密

所以通常更强调：

- OIDC / OAuth2 标准 flow
- PKCE
- 不随便暴露敏感凭证
- 必要时走 BFF

------

## 2）BFF / 服务端客户端

特点：

- 可以安全保存秘密
- 可以访问 JKS / 私钥 / Vault
- 更适合做 confidential client

所以很多服务端会：

- 用 JWT assertion
- 用 mTLS
- 在服务端拿 token
- 把浏览器隔离出去

------

## 3）纯后端服务到服务

特点：

- 不代表用户
- 可能只代表 app 自己

这时常见的是：

- client authentication
- app-to-app token
- short-lived access token

------

# 十、这章和你前面 rules 的关系

现在开始把概念和 rules 接上。

------

## API0-R2

这是这一章最核心的相关 rule。

它本质上在要求：

- 请求 API 时，要用 OIDC / OAuth2 标准
- 浏览器和 BFF 场景要处理正确
- 凭证要放对地方

------

## API2-R1

它在问：

- 不同类型 client
- 不同数据类型
- 不同场景

到底该用什么 flow。

------

## API2-R6

它在问：

- Authorization Code Grant 细节是否安全
- redirect_uri
- PKCE
- native app user-agent 选择

------

## API2-R7

它在问：

- confidential client 怎么证明自己是自己
- 用 JWT auth 还是 mTLS

------

## API2-R2 / API2-R3 / API2-R5

这些是在 OAuth/OIDC 已经建立后，进一步追问：

- token 活多久
- token 是什么格式
- token 怎么存、怎么撤销

------

## API2-R9

它和 OIDC 里的认证强度很相关，比如：

- MFA
- authentication method
- acr / amr

------

## API6-R1 / API7-R5

它们会跟：

- nonce
- state
- replay
- CSRF
   这些 OIDC/OAuth 细节强相关。

------

# 十一、你以后看项目时，哪些现象说明它在用 OAuth2 / OIDC

这个很实用。

------

## 1）你会在配置里看到

- `client_id`
- token endpoint
- authorization endpoint
- issuer
- redirect_uri
- scope
- PKCE
- JWK / public key
- audience

------

## 2）你会在代码里看到

- `Authorization: Bearer ...`
- 调 token endpoint
- auth callback
- OAuth/OIDC library
- ID token parsing
- access token validation

------

## 3）你会在架构图里看到

- WebSSO
- OIDC provider
- OAuth server
- Apigee + token validation
- Browser / BFF / API flow

------

# 十二、一个完整的小例子，把这一章串起来

假设用户用 Angular 前端登录系统。

### 第一步

Angular 把用户带去 WebSSO 登录。
 这里开始进入 OIDC / OAuth2 流程。

### 第二步

WebSSO 认证用户。
 如果成功，会签发相应结果。

### 第三步

客户端按某个 flow 拿到 token。
 通常至少会有 access token；如果是 OIDC，还常会有 ID token。

### 第四步

Angular 或 BFF 拿 access token 去调 Apigee。

### 第五步

Apigee / backend 根据 token 判断：

- token 是否有效
- scope 是否够
- 该不该让请求继续走

你看到没有：

- OAuth2 / OIDC 不只是“登录页”
- 它贯穿了从登录到 API 调用整个链路

------

# 十三、这一章最重要的 8 句话

## 第一句

**OAuth 2.0 更偏“怎么获得访问 API 的许可”。**

## 第二句

**OIDC 在 OAuth 2.0 之上，更偏“怎么标准化表达用户身份认证结果”。**

## 第三句

**OAuth2 不等于 OIDC，OIDC 也不是脱离 OAuth2 单独存在的。**

## 第四句

**现代 API 项目常同时需要两者：既要知道用户是谁，也要知道怎么安全地访问 API。**

## 第五句

**Client 通过某种 grant flow 拿 token，不同 client 适合不同 flow。**

## 第六句

**Access token 更偏给 API 用，ID token 更偏给客户端理解登录结果用。**

## 第七句

**浏览器、BFF、后端服务使用 OAuth2/OIDC 的方式不会完全一样。**

## 第八句

**你做 assessment 时，不是只看“项目用了 OAuth”，而是看“用得对不对、用在哪一层、flow 是否合适”。**

------

## 十四、一句话总结这章

你现在可以先把 OAuth2 / OIDC 理解成：
 **现代 API 系统里，用来标准化“登录、发 token、访问 API”这一整条链路的基础框架。**



# 第 5 部分：Grant Flow——token 到底是怎么拿到的

这一章你先抓住一句话：

> **Grant Flow 就是 Client 获取 token 的路径和方式。**

你可以把它理解成：

- 不是所有客户端都用同一条路拿 token
- 不同场景适合不同 flow
- 选错 flow，本身就是安全问题

------

## 一、先解释什么叫 Grant Flow

### 1）最通俗的定义

**Grant Flow** 就是：

> 客户端按照什么步骤，从授权服务器 / OpenID Provider 那里拿到 token。

### 2）为什么叫 grant

因为从授权视角看，它在说的是：

> 系统怎么把“访问许可”授予给客户端。

### 3）为什么它这么重要

因为 flow 决定了很多关键问题：

- 是不是有用户参与
- 是不是适合浏览器
- 客户端能不能安全保存秘密
- token 是不是容易被截获
- 是否需要额外保护（比如 PKCE）

------

## 二、先把拿 token 的场景分成 3 大类

这是最重要的第一步。
 你以后看到任何项目，先别急着想 flow 名字，先判断它属于哪类场景。

------

### 场景 1：代表用户访问 API

也就是说：

- 用户是资源所有者
- client 代表 user 去调 API

### 例子

- Angular 前端代表客户查账户
- 手机 App 代表用户看订单
- BFF 代表已登录用户调用后端 API

### 特点

- 需要用户登录
- 往往和 OIDC 强相关
- 常常要同时拿到用户身份信息和 API 访问许可

### 相关 rules

- API0-R2
- API2-R1
- API2-R6
- API2-R8
- API2-R9

------

### 场景 2：客户端代表自己访问 API

也就是说：

- 不涉及最终用户
- 只是 app 自己以 app 身份访问 API

### 例子

- batch job 调内部 API
- BFF 启动时调用配置服务
- Spring 服务调用 APIM token endpoint
- 一个内部服务访问另一个服务

### 特点

- 没有用户登录页面
- 更偏 client authentication
- 常见于 confidential client

### 相关 rules

- API2-R1
- API2-R7
- API2-R2
- API2-R3

------

### 场景 3：设备或特殊终端访问

也就是说：

- 客户端不适合正常打开登录页
- 输入能力有限
- 设备本身不方便走标准浏览器交互

### 例子

- 智能电视
- 特殊终端
- 某些设备类应用

### 相关 rules

- API2-R1
- API2-R2
- API2-R5

------

# 三、为什么不能“所有 client 都统一一种 flow”

这是很多小白最容易误解的点。

你可能会想：

> “既然 OAuth2/OIDC 很标准，那所有客户端都用一种不就好了？”

不行。因为不同 client 的安全能力完全不同。

------

## 1）浏览器前端的问题

浏览器前端是 **public client**，它的问题是：

- 代码会到用户设备
- 不适合保存长期秘密
- URL 和前端环境更容易被观察
- 浏览器跳转和回调容易受前端场景影响

所以浏览器前端不能随便拿敏感凭证，也不能乱用不适合它的 flow。

------

## 2）服务端客户端的问题

服务端是 **confidential client**，它通常：

- 能保存 secret / private key
- 能读 JKS / Vault
- 能做服务端 token 获取
- 更适合做更强的客户端认证

所以它可以用一些浏览器不适合用的方式。

------

## 3）设备终端的问题

设备可能：

- 不方便输入用户名密码
- 不方便跳浏览器
- 不方便完成复杂交互

所以它要走专门的 device flow。

------

# 四、先建立一个总图：你后面最常见的 4 个 flow

你现在先只记这 4 个，够你当前 assessment 用了。

1. **Authorization Code Grant**
2. **Authorization Code Grant + PKCE**
3. **Client Credentials**
4. **Device Code Grant**

后面你还会见到：

1. **Token Exchange**

------

# 五、Authorization Code Grant：最经典的“用户登录后换 token”流程

------

## 1）它是什么

### 小白版定义

用户先去登录，登录完成后客户端拿到一个短暂的 **authorization code**，然后再用这个 code 去换 token。

### 通俗理解

你可以把它理解成：

> **先拿一个一次性的临时兑换券，再拿兑换券去换真正的 token。**

------

## 2）为什么不直接一步发 access token

因为如果直接把 access token 暴露在前端跳转链路里，风险更大。
 而 authorization code 是：

- 短时有效
- 一次性使用
- 只是中间凭证

这样比直接暴露 access token 更安全。

------

## 3）大致流程长什么样

我们先不讲特别细的参数，你先看大图。

### 第一步

Client 把用户带去 WebSSO / Authorization Server 登录。

### 第二步

用户完成认证。

### 第三步

Authorization Server 把用户带回 client 的回调地址，并附带一个 **authorization code**。

### 第四步

Client 再拿这个 code 去调用 token endpoint，换取：

- access token
- 有时还有 ID token
- 有时还有 refresh token

------

## 4）适合哪些场景

最经典的用户登录型场景，特别适合：

- 有用户参与
- 需要标准登录
- 需要回调跳转

### 企业项目里常见

- Web 应用
- BFF 场景
- 某些 mobile/native app（配合 PKCE）

------

## 5）它和 rules 的关系

### 强相关

- API2-R1
- API2-R6
- API0-R2

### 为什么

因为这些 rule 在问：

- 你是不是用了正确 flow
- redirect_uri 是否安全
- auth code 是否安全
- browser/native app 是否用对方式

------

# 六、Authorization Code Grant + PKCE：为什么现在几乎一定会提 PKCE

------

## 1）先说 PKCE 是什么

### 小白版定义

**PKCE** 是给 Authorization Code 流程额外加的一层保护，防止 authorization code 被截获后被别人拿去换 token。

### 通俗理解

你可以把 PKCE 理解成：

> “就算别人偷到了 code，没有我手里的那把配对钥匙，也换不出 token。”

------

## 2）为什么 Authorization Code 还不够

因为 code 本身虽然是中间凭证，但如果被截获，攻击者还是可能尝试拿它去换 token。

尤其在这些场景里风险更高：

- public client
- 浏览器前端
- native app
- 用户设备环境不完全可信

------

## 3）PKCE 大致怎么工作

这里先讲直觉，不讲数学。

### 第一步

Client 在一开始自己生成一个随机字符串，叫 **code_verifier**。

### 第二步

Client 再把它加工成一个 **code_challenge**，发给授权服务器。

### 第三步

用户正常登录，拿回 authorization code。

### 第四步

Client 去换 token 时，除了 code，还要提交最初那个 **code_verifier**。

### 第五步

服务器检查：

- 你现在提交的 verifier
- 是不是和一开始登记的 challenge 对得上

如果对不上，就不给 token。

------

## 4）你现在先怎么记 code_verifier 和 code_challenge

不用怕这两个名字。

### code_verifier

你可以理解成：

- 原始秘密值
- client 自己留着，不先暴露出去

### code_challenge

你可以理解成：

- verifier 的“加工后版本”
- 先交给授权服务器备案

------

## 5）为什么 PKCE 对 public client 特别重要

因为 public client 保不住 secret。
 所以不能指望：

- client_secret
- 强保密环境

那就要靠 PKCE 这种机制，给 auth code 流程多一层保护。

------

## 6）在哪些 rule 里特别重要

- **API2-R1**
- **API2-R6**
- **API0-R2**

尤其你前面的 rule 已经明确说：

- public client 请求用户数据时，要走 Auth Code + PKCE
- untrusted client 要用 PKCE
- challenge method 要用 `S256`

------

# 七、Client Credentials：没有用户，只有应用自己

------

## 1）它是什么

### 小白版定义

**Client Credentials** 是指：
 客户端不代表用户，只代表自己，以自己的应用身份去拿 token。

### 通俗理解

你可以把它理解成：

> “不是张三在调用 API，而是这个系统自己在调用 API。”

------

## 2）什么时候会用它

当调用场景里没有 user 时。

### 典型例子

- batch job 拉数据
- 服务 A 调服务 B
- 一个后台进程调用内部接口
- 一个服务去拿 APIM 的 token 再调别的 API

------

## 3）为什么它和浏览器前端不一样

因为它通常发生在服务端：

- confidential client
- 可以做客户端认证
- 可以持有 key / certificate / secret

所以它适合更强的 app-to-app 场景。

------

## 4）大致流程

### 第一步

Client 向授权服务器证明自己是哪个应用。

### 第二步

授权服务器验证这个 client。

### 第三步

如果通过，就发 access token。

这里通常没有：

- 用户登录页
- 用户 consent
- 用户身份上下文

------

## 5）它和 user-based flow 最本质的区别

### User-based flow

在问：

- 这个用户授权这个 app 访问什么

### Client Credentials

在问：

- 这个 app 自己被允许访问什么

------

## 6）和 rules 的关系

- **API2-R1**
- **API2-R7**
- **API2-R2**
- **API2-R3**

因为你会进一步关心：

- app 是怎么认证自己的
- token 寿命多长
- token 格式是什么
- scope 怎么给

------

# 八、Device Code Grant：给“不会正常登录”的设备准备的

------

## 1）它是什么

### 小白版定义

一种给输入能力差、浏览器能力差的设备准备的授权流程。

### 通俗理解

设备自己不方便完成完整登录，于是让用户在另一个更方便的设备上完成授权。

------

## 2）典型例子

- 智能电视
- 某些终端设备
- 输入能力弱的装置

------

## 3）大致流程

### 第一步

设备向授权服务器申请 device flow，拿到：

- device code
- user code

### 第二步

设备把 user code 显示给用户。

### 第三步

用户在另一台更方便的设备上（比如手机/浏览器）去完成认证，并输入 user code。

### 第四步

设备轮询（polling）授权服务器，看授权是否已完成。

### 第五步

完成后拿到 token。

------

## 4）它为什么在 rules 里出现

因为这是标准 flow 之一，但它有特殊风险点：

- device code 生命周期
- user code 强度
- polling 间隔
- refresh token 怎么发

所以你前面的 rules 才会专门列它。

------

# 九、Token Exchange：用一个 token 去换另一个 token

这个你当前不需要理解得太深，但要先有个感觉。

------

## 1）它是什么

### 小白版定义

客户端拿着一个已有 token，再去换一个更适合当前用途的新 token。

### 通俗理解

你可以理解成：

> “把原来那张大门票，换成一张只适合这个下游场景的小门票。”

------

## 2）为什么会需要它

因为一个 token 未必适合所有场景。

比如：

- 原 token audience 太大
- 原 token 生命周期太长
- 原 token 权限太宽
- 下游系统只接受特定形式 token

于是可以换一张：

- 更短命
- 更窄 scope
- 特定 audience
- 更适合内部调用

------

## 3）和企业项目的关系

在复杂微服务和网关场景里，它可以帮助做：

- 最小权限收缩
- 下游 audience 控制
- 安全边界隔离

------

# 十、Redirect URI：为什么 rules 会专门管它

这个概念和 Authorization Code 流程强相关。

------

## 1）它是什么

**redirect_uri** 就是：

> 登录完成后，授权服务器把用户带回去的地址。

------

## 2）为什么重要

因为如果 redirect_uri 控制不好，攻击者可能让：

- code
- token
- 登录结果

被送到错误地址。

这相当于把认证结果送错人。

------

## 3）企业里为什么要严格控制

所以 rule 会要求：

- fully-qualified
- 预注册
- 不能 wildcard
- 客户端和授权服务器都要校验

------

## 4）你在 assessment 里怎么用

看到以下内容时想到 redirect_uri：

- OIDC 登录配置
- WebSSO client registration
- application.yml / frontend config
- redirect callback endpoint

------

# 十一、为什么“选错 flow”本身就是安全问题

这一点你一定要提前建立起来。

------

## 1）浏览器用了不适合浏览器的 flow

风险：

- 凭证容易暴露
- code/token 容易被截
- 无法满足 public client 约束

------

## 2）服务端没有用足够强的 client authentication

风险：

- app 身份不可靠
- token 申请过程容易被伪造

------

## 3）本该代表用户的请求，却只用 app 自己身份

风险：

- 后端不知道真实 user
- 审计/授权边界错位

------

## 4）高敏感场景还在用过弱 flow

风险：

- token 太容易被滥用
- 不符合企业规则

------

# 十二、你在项目里，去哪里看 flow

这部分很实用。

------

## 1）看架构图

看：

- 用户是不是会被重定向到 WebSSO
- 浏览器是不是直接拿 token
- 有没有 BFF
- 服务之间是不是自己拿 token

------

## 2）看前端代码

找：

- login redirect
- callback route
- PKCE library
- code / token parsing
- redirect_uri 配置

------

## 3）看后端代码

找：

- token endpoint 调用
- `grant_type`
- `assertion`
- client authentication
- `RestTemplate` / `WebClient` 拿 token 逻辑

------

## 4）看配置

找：

- `client_id`
- authorization endpoint
- token endpoint
- redirect URI
- scope
- issuer
- key/certificate config

------

## 5）看 WebSSO / IdP 配置

找：

- client registration
- allowed grant types
- redirect URIs
- PKCE required?
- confidential or public client?
- refresh token policy

------

# 十三、这一章和 rules 的对应关系

------

## API2-R1 是最核心的一条

它直接在问：

- 这个 client 是什么类型
- 请求的是哪类数据
- 有没有 user 参与
- 应该用哪种 flow

------

## API2-R6

它会继续追问 Authorization Code Grant 的安全细节：

- redirect_uri
- user-agent
- PKCE

------

## API0-R2

它从更高层要求：

- API access must use OIDC/OAuth2 standards
- browser/BFF 场景要处理正确

------

## API3-R4

会从 least privilege 角度问：

- client 有没有只选必要的 flow

------

# 十四、一个完整例子，把这章串起来

假设一个 Angular 前端访问客户 API。

### 情况 A：Angular 代表用户访问

那通常应该走：

- OIDC / OAuth
- Authorization Code + PKCE

原因：

- 有 user
- 是 public client
- 需要标准登录流程
- 要防 code 被截

------

### 情况 B：Spring BFF 代表自己先拿 token

那可能走：

- confidential client 的 app-to-app flow
- 或某种 assertion / client auth 模式

原因：

- 没有浏览器直接参与 token 申请
- 服务端能保存秘密
- 适合更强认证

------

### 情况 C：智能设备登录

那可能走：

- Device Code Grant

原因：

- 设备不适合正常浏览器登录

------

# 十五、这一章最重要的 8 句话

## 第一句

**Grant Flow 就是 Client 如何拿到 token 的方式。**

## 第二句

**不是所有 client 都适合同一种 flow。**

## 第三句

**有用户参与和没有用户参与，是决定 flow 的第一分界线。**

## 第四句

**Authorization Code Grant 是经典用户登录后换 token 的流程。**

## 第五句

**PKCE 是给 Authorization Code 流程加的一层防截码保护，尤其适合 public client。**

## 第六句

**Client Credentials 是 app 代表自己，不代表用户。**

## 第七句

**Redirect URI 如果控制不好，认证结果可能被送错地方。**

## 第八句

**做 assessment 时，不是只看有没有 token，而是看 token 是怎么拿到的。**

------

## 十六、一句话总结这章

你现在可以先把 flow 理解成：
 **不同类型的客户端，根据自己是否代表用户、是否能保住秘密、是否适合浏览器交互，走不同路径去拿 token。**



# 第 5 部分：Grant Flow——token 到底是怎么拿到的

这一章你先抓住一句话：

> **Grant Flow 就是 Client 获取 token 的路径和方式。**

你可以把它理解成：

- 不是所有客户端都用同一条路拿 token
- 不同场景适合不同 flow
- 选错 flow，本身就是安全问题

------

## 一、先解释什么叫 Grant Flow

### 1）最通俗的定义

**Grant Flow** 就是：

> 客户端按照什么步骤，从授权服务器 / OpenID Provider 那里拿到 token。

### 2）为什么叫 grant

因为从授权视角看，它在说的是：

> 系统怎么把“访问许可”授予给客户端。

### 3）为什么它这么重要

因为 flow 决定了很多关键问题：

- 是不是有用户参与
- 是不是适合浏览器
- 客户端能不能安全保存秘密
- token 是不是容易被截获
- 是否需要额外保护（比如 PKCE）

------

## 二、先把拿 token 的场景分成 3 大类

这是最重要的第一步。
 你以后看到任何项目，先别急着想 flow 名字，先判断它属于哪类场景。

------

### 场景 1：代表用户访问 API

也就是说：

- 用户是资源所有者
- client 代表 user 去调 API

### 例子

- Angular 前端代表客户查账户
- 手机 App 代表用户看订单
- BFF 代表已登录用户调用后端 API

### 特点

- 需要用户登录
- 往往和 OIDC 强相关
- 常常要同时拿到用户身份信息和 API 访问许可

### 相关 rules

- API0-R2
- API2-R1
- API2-R6
- API2-R8
- API2-R9

------

### 场景 2：客户端代表自己访问 API

也就是说：

- 不涉及最终用户
- 只是 app 自己以 app 身份访问 API

### 例子

- batch job 调内部 API
- BFF 启动时调用配置服务
- Spring 服务调用 APIM token endpoint
- 一个内部服务访问另一个服务

### 特点

- 没有用户登录页面
- 更偏 client authentication
- 常见于 confidential client

### 相关 rules

- API2-R1
- API2-R7
- API2-R2
- API2-R3

------

### 场景 3：设备或特殊终端访问

也就是说：

- 客户端不适合正常打开登录页
- 输入能力有限
- 设备本身不方便走标准浏览器交互

### 例子

- 智能电视
- 特殊终端
- 某些设备类应用

### 相关 rules

- API2-R1
- API2-R2
- API2-R5

------

# 三、为什么不能“所有 client 都统一一种 flow”

这是很多小白最容易误解的点。

你可能会想：

> “既然 OAuth2/OIDC 很标准，那所有客户端都用一种不就好了？”

不行。因为不同 client 的安全能力完全不同。

------

## 1）浏览器前端的问题

浏览器前端是 **public client**，它的问题是：

- 代码会到用户设备
- 不适合保存长期秘密
- URL 和前端环境更容易被观察
- 浏览器跳转和回调容易受前端场景影响

所以浏览器前端不能随便拿敏感凭证，也不能乱用不适合它的 flow。

------

## 2）服务端客户端的问题

服务端是 **confidential client**，它通常：

- 能保存 secret / private key
- 能读 JKS / Vault
- 能做服务端 token 获取
- 更适合做更强的客户端认证

所以它可以用一些浏览器不适合用的方式。

------

## 3）设备终端的问题

设备可能：

- 不方便输入用户名密码
- 不方便跳浏览器
- 不方便完成复杂交互

所以它要走专门的 device flow。

------

# 四、先建立一个总图：你后面最常见的 4 个 flow

你现在先只记这 4 个，够你当前 assessment 用了。

1. **Authorization Code Grant**
2. **Authorization Code Grant + PKCE**
3. **Client Credentials**
4. **Device Code Grant**

后面你还会见到：

1. **Token Exchange**

------

# 五、Authorization Code Grant：最经典的“用户登录后换 token”流程

------

## 1）它是什么

### 小白版定义

用户先去登录，登录完成后客户端拿到一个短暂的 **authorization code**，然后再用这个 code 去换 token。

### 通俗理解

你可以把它理解成：

> **先拿一个一次性的临时兑换券，再拿兑换券去换真正的 token。**

------

## 2）为什么不直接一步发 access token

因为如果直接把 access token 暴露在前端跳转链路里，风险更大。
 而 authorization code 是：

- 短时有效
- 一次性使用
- 只是中间凭证

这样比直接暴露 access token 更安全。

------

## 3）大致流程长什么样

我们先不讲特别细的参数，你先看大图。

### 第一步

Client 把用户带去 WebSSO / Authorization Server 登录。

### 第二步

用户完成认证。

### 第三步

Authorization Server 把用户带回 client 的回调地址，并附带一个 **authorization code**。

### 第四步

Client 再拿这个 code 去调用 token endpoint，换取：

- access token
- 有时还有 ID token
- 有时还有 refresh token

------

## 4）适合哪些场景

最经典的用户登录型场景，特别适合：

- 有用户参与
- 需要标准登录
- 需要回调跳转

### 企业项目里常见

- Web 应用
- BFF 场景
- 某些 mobile/native app（配合 PKCE）

------

## 5）它和 rules 的关系

### 强相关

- API2-R1
- API2-R6
- API0-R2

### 为什么

因为这些 rule 在问：

- 你是不是用了正确 flow
- redirect_uri 是否安全
- auth code 是否安全
- browser/native app 是否用对方式

------

# 六、Authorization Code Grant + PKCE：为什么现在几乎一定会提 PKCE

------

## 1）先说 PKCE 是什么

### 小白版定义

**PKCE** 是给 Authorization Code 流程额外加的一层保护，防止 authorization code 被截获后被别人拿去换 token。

### 通俗理解

你可以把 PKCE 理解成：

> “就算别人偷到了 code，没有我手里的那把配对钥匙，也换不出 token。”

------

## 2）为什么 Authorization Code 还不够

因为 code 本身虽然是中间凭证，但如果被截获，攻击者还是可能尝试拿它去换 token。

尤其在这些场景里风险更高：

- public client
- 浏览器前端
- native app
- 用户设备环境不完全可信

------

## 3）PKCE 大致怎么工作

这里先讲直觉，不讲数学。

### 第一步

Client 在一开始自己生成一个随机字符串，叫 **code_verifier**。

### 第二步

Client 再把它加工成一个 **code_challenge**，发给授权服务器。

### 第三步

用户正常登录，拿回 authorization code。

### 第四步

Client 去换 token 时，除了 code，还要提交最初那个 **code_verifier**。

### 第五步

服务器检查：

- 你现在提交的 verifier
- 是不是和一开始登记的 challenge 对得上

如果对不上，就不给 token。

------

## 4）你现在先怎么记 code_verifier 和 code_challenge

不用怕这两个名字。

### code_verifier

你可以理解成：

- 原始秘密值
- client 自己留着，不先暴露出去

### code_challenge

你可以理解成：

- verifier 的“加工后版本”
- 先交给授权服务器备案

------

## 5）为什么 PKCE 对 public client 特别重要

因为 public client 保不住 secret。
 所以不能指望：

- client_secret
- 强保密环境

那就要靠 PKCE 这种机制，给 auth code 流程多一层保护。

------

## 6）在哪些 rule 里特别重要

- **API2-R1**
- **API2-R6**
- **API0-R2**

尤其你前面的 rule 已经明确说：

- public client 请求用户数据时，要走 Auth Code + PKCE
- untrusted client 要用 PKCE
- challenge method 要用 `S256`

------

# 七、Client Credentials：没有用户，只有应用自己

------

## 1）它是什么

### 小白版定义

**Client Credentials** 是指：
 客户端不代表用户，只代表自己，以自己的应用身份去拿 token。

### 通俗理解

你可以把它理解成：

> “不是张三在调用 API，而是这个系统自己在调用 API。”

------

## 2）什么时候会用它

当调用场景里没有 user 时。

### 典型例子

- batch job 拉数据
- 服务 A 调服务 B
- 一个后台进程调用内部接口
- 一个服务去拿 APIM 的 token 再调别的 API

------

## 3）为什么它和浏览器前端不一样

因为它通常发生在服务端：

- confidential client
- 可以做客户端认证
- 可以持有 key / certificate / secret

所以它适合更强的 app-to-app 场景。

------

## 4）大致流程

### 第一步

Client 向授权服务器证明自己是哪个应用。

### 第二步

授权服务器验证这个 client。

### 第三步

如果通过，就发 access token。

这里通常没有：

- 用户登录页
- 用户 consent
- 用户身份上下文

------

## 5）它和 user-based flow 最本质的区别

### User-based flow

在问：

- 这个用户授权这个 app 访问什么

### Client Credentials

在问：

- 这个 app 自己被允许访问什么

------

## 6）和 rules 的关系

- **API2-R1**
- **API2-R7**
- **API2-R2**
- **API2-R3**

因为你会进一步关心：

- app 是怎么认证自己的
- token 寿命多长
- token 格式是什么
- scope 怎么给

------

# 八、Device Code Grant：给“不会正常登录”的设备准备的

------

## 1）它是什么

### 小白版定义

一种给输入能力差、浏览器能力差的设备准备的授权流程。

### 通俗理解

设备自己不方便完成完整登录，于是让用户在另一个更方便的设备上完成授权。

------

## 2）典型例子

- 智能电视
- 某些终端设备
- 输入能力弱的装置

------

## 3）大致流程

### 第一步

设备向授权服务器申请 device flow，拿到：

- device code
- user code

### 第二步

设备把 user code 显示给用户。

### 第三步

用户在另一台更方便的设备上（比如手机/浏览器）去完成认证，并输入 user code。

### 第四步

设备轮询（polling）授权服务器，看授权是否已完成。

### 第五步

完成后拿到 token。

------

## 4）它为什么在 rules 里出现

因为这是标准 flow 之一，但它有特殊风险点：

- device code 生命周期
- user code 强度
- polling 间隔
- refresh token 怎么发

所以你前面的 rules 才会专门列它。

------

# 九、Token Exchange：用一个 token 去换另一个 token

这个你当前不需要理解得太深，但要先有个感觉。

------

## 1）它是什么

### 小白版定义

客户端拿着一个已有 token，再去换一个更适合当前用途的新 token。

### 通俗理解

你可以理解成：

> “把原来那张大门票，换成一张只适合这个下游场景的小门票。”

------

## 2）为什么会需要它

因为一个 token 未必适合所有场景。

比如：

- 原 token audience 太大
- 原 token 生命周期太长
- 原 token 权限太宽
- 下游系统只接受特定形式 token

于是可以换一张：

- 更短命
- 更窄 scope
- 特定 audience
- 更适合内部调用

------

## 3）和企业项目的关系

在复杂微服务和网关场景里，它可以帮助做：

- 最小权限收缩
- 下游 audience 控制
- 安全边界隔离

------

# 十、Redirect URI：为什么 rules 会专门管它

这个概念和 Authorization Code 流程强相关。

------

## 1）它是什么

**redirect_uri** 就是：

> 登录完成后，授权服务器把用户带回去的地址。

------

## 2）为什么重要

因为如果 redirect_uri 控制不好，攻击者可能让：

- code
- token
- 登录结果

被送到错误地址。

这相当于把认证结果送错人。

------

## 3）企业里为什么要严格控制

所以 rule 会要求：

- fully-qualified
- 预注册
- 不能 wildcard
- 客户端和授权服务器都要校验

------

## 4）你在 assessment 里怎么用

看到以下内容时想到 redirect_uri：

- OIDC 登录配置
- WebSSO client registration
- application.yml / frontend config
- redirect callback endpoint

------

# 十一、为什么“选错 flow”本身就是安全问题

这一点你一定要提前建立起来。

------

## 1）浏览器用了不适合浏览器的 flow

风险：

- 凭证容易暴露
- code/token 容易被截
- 无法满足 public client 约束

------

## 2）服务端没有用足够强的 client authentication

风险：

- app 身份不可靠
- token 申请过程容易被伪造

------

## 3）本该代表用户的请求，却只用 app 自己身份

风险：

- 后端不知道真实 user
- 审计/授权边界错位

------

## 4）高敏感场景还在用过弱 flow

风险：

- token 太容易被滥用
- 不符合企业规则

------

# 十二、你在项目里，去哪里看 flow

这部分很实用。

------

## 1）看架构图

看：

- 用户是不是会被重定向到 WebSSO
- 浏览器是不是直接拿 token
- 有没有 BFF
- 服务之间是不是自己拿 token

------

## 2）看前端代码

找：

- login redirect
- callback route
- PKCE library
- code / token parsing
- redirect_uri 配置

------

## 3）看后端代码

找：

- token endpoint 调用
- `grant_type`
- `assertion`
- client authentication
- `RestTemplate` / `WebClient` 拿 token 逻辑

------

## 4）看配置

找：

- `client_id`
- authorization endpoint
- token endpoint
- redirect URI
- scope
- issuer
- key/certificate config

------

## 5）看 WebSSO / IdP 配置

找：

- client registration
- allowed grant types
- redirect URIs
- PKCE required?
- confidential or public client?
- refresh token policy

------

# 十三、这一章和 rules 的对应关系

------

## API2-R1 是最核心的一条

它直接在问：

- 这个 client 是什么类型
- 请求的是哪类数据
- 有没有 user 参与
- 应该用哪种 flow

------

## API2-R6

它会继续追问 Authorization Code Grant 的安全细节：

- redirect_uri
- user-agent
- PKCE

------

## API0-R2

它从更高层要求：

- API access must use OIDC/OAuth2 standards
- browser/BFF 场景要处理正确

------

## API3-R4

会从 least privilege 角度问：

- client 有没有只选必要的 flow

------

# 十四、一个完整例子，把这章串起来

假设一个 Angular 前端访问客户 API。

### 情况 A：Angular 代表用户访问

那通常应该走：

- OIDC / OAuth
- Authorization Code + PKCE

原因：

- 有 user
- 是 public client
- 需要标准登录流程
- 要防 code 被截

------

### 情况 B：Spring BFF 代表自己先拿 token

那可能走：

- confidential client 的 app-to-app flow
- 或某种 assertion / client auth 模式

原因：

- 没有浏览器直接参与 token 申请
- 服务端能保存秘密
- 适合更强认证

------

### 情况 C：智能设备登录

那可能走：

- Device Code Grant

原因：

- 设备不适合正常浏览器登录

------

# 十五、这一章最重要的 8 句话

## 第一句

**Grant Flow 就是 Client 如何拿到 token 的方式。**

## 第二句

**不是所有 client 都适合同一种 flow。**

## 第三句

**有用户参与和没有用户参与，是决定 flow 的第一分界线。**

## 第四句

**Authorization Code Grant 是经典用户登录后换 token 的流程。**

## 第五句

**PKCE 是给 Authorization Code 流程加的一层防截码保护，尤其适合 public client。**

## 第六句

**Client Credentials 是 app 代表自己，不代表用户。**

## 第七句

**Redirect URI 如果控制不好，认证结果可能被送错地方。**

## 第八句

**做 assessment 时，不是只看有没有 token，而是看 token 是怎么拿到的。**

------

## 十六、一句话总结这章

你现在可以先把 flow 理解成：
 **不同类型的客户端，根据自己是否代表用户、是否能保住秘密、是否适合浏览器交互，走不同路径去拿 token。**

------

下一步我们进入 **第 6 部分：Token 全家桶——每种 token 分别干什么**。

继续

Thought for 9s

好，我们继续 **第 6 部分：Token 全家桶——每种 token 分别干什么**。

**直观结论：**
 你后面做 assessment 时，看到 `token` 这个词，绝对不能只停留在“这是个令牌”。
 因为 rules 里提到的 token 其实分很多种，它们的职责完全不同：
 **有的拿来调 API，有的拿来表示登录结果，有的拿来续命，有的只是中间兑换券，有的甚至只是客户端证明材料。**
 如果把这些混在一起，后面所有关于生命周期、存储、撤销、格式、泄露风险的判断都会乱掉。

------

# 第 6 部分：Token 全家桶——每种 token 分别干什么

这一章你先抓住一句话：

> **Token 不是一种东西，而是一整个家族。**

你后面看项目、看代码、看规则时，第一反应都应该是：

**这到底是哪一种 token，它是给谁用的，它在流程里扮演什么角色。**

------

## 一、先说最基础的：token 到底是什么

### 1）小白版定义

**Token** 可以先理解成：

> 系统签发的一种“临时凭证”，用来表示某种身份、权限或流程状态。

### 2）为什么不用密码直接搞定一切

因为 token 比密码更适合 API 场景：

- 可以短时有效
- 可以限制权限范围
- 可以撤销
- 可以区分不同 client
- 不需要让所有系统都接触用户密码

### 3）通俗比喻

你可以把 token 想成“临时通行证”。

但注意：不同 token 是不同类型的通行证：

- 有的是“进楼证”
- 有的是“身份证明”
- 有的是“续期卡”
- 有的是“兑换券”

------

## 二、先建立一张总图：你最常见的几种 token

你当前阶段，先记住这 8 个就够用了：

1. **Access Token**
2. **ID Token**
3. **Refresh Token**
4. **API Key**
5. **Authorization Code**
6. **Device Code**
7. **User Code**
8. **JWT Authentication Token / Client Assertion**

另外还有一个你会反复看到但不属于“具体 token 种类”的词：

1. **Bearer**

它更像“token 的使用方式”，不是 token 家族成员本身。这个我们后面也会讲。

------

# 三、Access Token：真正拿去调 API 的通行证

------

## 1）它是什么

### 小白版定义

**Access Token** 是客户端真正拿去调用 API 的 token。

### 通俗理解

你可以把它理解成：

> “拿着它，你就能去请求某些 API。”

------

## 2）它是给谁看的

Access Token 主要是给：

- API Gateway
- API Provider
- Resource Server
- 后端业务服务

看的，不是主要给前端自己“读内容”用的。

------

## 3）它通常在什么时候出现

典型场景是：

1. Client 先完成认证/授权流程
2. Authorization Server / OpenID Provider 发出 access token
3. Client 拿它放到请求里
4. API 收到后进行校验

最常见的样子就是：

```
Authorization: Bearer <access_token>
```

------

## 4）它的核心职责

它通常回答这些问题：

- 这个请求是谁发起的 / 代表谁发起的
- 这个请求是否还有效
- 这个请求允许访问哪些资源（scope）
- 这个 token 是否签发给当前目标 API

------

## 5）为什么它这么重要

因为它是最直接的“API 通行证”。

一旦 access token 泄露，攻击者就可能在有效期内直接用它调用 API。

所以 rules 才会反复关注：

- 生命周期是否够短
- 是否会出现在 URL / 日志
- 是否是 Bearer
- 是否需要撤销
- 是否适合 public client
- 是否只带必要 scope

------

## 6）你在项目里怎么认它

看到这些就高度怀疑是 access token：

- `Authorization: Bearer ...`
- `getAccessToken()`
- `access_token`
- gateway 上的 VerifyJWT / OAuth token verify
- downstream 调用前加 token header

------

## 7）涉及哪些 rules

- **API2-R2**（lifetime）
- **API2-R3**（format）
- **API2-R5**（management）
- **API1-R1**（authorization context）
- **API5-R1**（scope）
- **API6-R1**（replay risk）
- **API0-R2**

------

# 四、ID Token：告诉客户端“登录结果是谁”的 token

------

## 1）它是什么

### 小白版定义

**ID Token** 是 OIDC 里专门用来表达用户身份认证结果的 token。

### 通俗理解

你可以把它理解成：

> “这次登录成功了，登录的人是谁、什么时候登录的、用了什么认证方式，这份说明书就写在 ID Token 里。”

------

## 2）它是给谁看的

和 access token 不一样，ID Token 主要是给：

- Client

看的。

也就是说，它主要服务于客户端理解“这次登录结果”。

------

## 3）它不应该主要拿来干嘛

它**不应该**作为主要 API 调用凭证去访问业务 API。

这是一个特别容易犯的错误。

### 错误理解

“我已经有一个 token 了，那就拿去调 API 吧。”

### 正确理解

- **ID Token**：告诉客户端用户是谁
- **Access Token**：拿去调 API

------

## 4）为什么 OIDC 需要它

因为 OAuth2 本身更偏“访问许可”，不够标准化地表达用户认证结果。
 所以 OIDC 增加了 ID Token，解决：

- 用户是谁
- 认证何时发生
- 认证强度如何
- 登录是不是刚刚完成的

------

## 5）你在项目里怎么认它

看到这些就要想到 ID token：

- `id_token`
- OIDC callback response
- 登录后解析用户身份
- `acr` / `amr` / `auth_time`
- 前端拿来显示用户资料或判断认证上下文

------

## 6）和 access token 的一句话区别

这个你一定先记死：

> **Access Token 是给 API 用的，ID Token 是给 Client 用的。**

------

## 7）涉及哪些 rules

- **API2-R2**
- **API2-R3**
- **API2-R5**
- **API2-R9**
- **API6-R1**

------

# 五、Refresh Token：拿来“续命”的 token

------

## 1）它是什么

### 小白版定义

**Refresh Token** 是用来换取新的 access token 的 token。

### 通俗理解

你可以把它理解成：

> “旧的 access token 快过期了，用这张续期卡去换一张新的。”

------

## 2）为什么需要它

因为 access token 通常应该短时有效。
 但如果每次过期都逼用户完整重新登录，体验会很差。

所以很多系统会设计成：

- access token：短命
- refresh token：稍长一些，用于续命

------

## 3）为什么 rules 对它特别敏感

因为 refresh token 往往活得更久，所以一旦泄露，风险比 access token 更大。

所以企业规则会很严格地问：

- 哪类 client 可以有 refresh token
- 浏览器能不能存 refresh token
- refresh token 是否必须 rotation
- 是否 one-time use
- 是否可以撤销
- 最多有多少 active refresh token

------

## 4）为什么浏览器场景经常容易出问题

因为浏览器属于 public client，保存长期敏感凭证风险大。

这也是为什么你前面的 rules 会对 browser-based application 下的 refresh token 特别敏感，甚至很多场景直接禁止。

------

## 5）你在项目里怎么认它

看到这些就想到 refresh token：

- `refresh_token`
- token refresh endpoint
- silent refresh
- rotation
- renew token
- “access token 过期后自动续签”

------

## 6）涉及哪些 rules

- **API2-R2**
- **API2-R5**
- **API2-R8**
- **API0-R2**

------

# 六、API Key：更像“应用标识型密钥”

------

## 1）它是什么

### 小白版定义

**API Key** 是一种用来标识调用方应用的密钥。

### 通俗理解

你可以先把它理解成：

> “这是哪个应用在调用我”的一种较简单证明方式。

------

## 2）它和 access token 的区别

这是很重要的一点。

### Access Token

更适合表达：

- 动态访问许可
- scope
- user/client 上下文
- 短时可控

### API Key

更像：

- 调用方应用标识
- 基础调用识别
- 简单流量归属

------

## 3）为什么 rules 不太信任它

因为 API key 通常比较“粗”。

它不太适合：

- 高敏感用户授权
- 严格代表最终用户
- 复杂动态权限控制

所以 rules 会强调：

- API key 不能通过 URL 传
- API key 不能拿来换 access token
- API key 最多只是某种基础追踪或标识机制

------

## 4）你在项目里怎么认它

看到这些就想到 API key：

- `x-api-key`
- `apiKey`
- request header 里固定应用密钥
- partner integration 用 key 区分调用方

------

## 5）涉及哪些 rules

- **API2-R2**
- **API2-R3**
- **API2-R5**
- **API2-R10**

------

# 七、Authorization Code：不是最终 token，而是中间兑换券

------

## 1）它是什么

### 小白版定义

**Authorization Code** 是 Authorization Code Grant 流程里短时有效的一次性中间凭证。

### 通俗理解

你可以把它理解成：

> “用户登录完成后，先给客户端一张临时兑换券，再让客户端拿这张券去换真正的 token。”

------

## 2）为什么它不是最终 access token

因为如果在浏览器跳转阶段就直接给 access token，风险更高。
 而 code 是：

- 短时有效
- 一次性使用
- 中间态凭证

安全性更容易控制。

------

## 3）为什么 rules 会特别强调它

因为它虽然只是中间件，但如果被截获，攻击者可能尝试用它去换 token。

所以规则会问：

- 它多久过期
- 是否 one-time use
- 是否和 client 绑定
- 是否配合 PKCE

------

## 4）你在项目里怎么认它

看到这些想到 authorization code：

- 登录回调 URL 里有 `code=...`
- 后端/前端拿 code 去请求 token endpoint
- `authorization_code` 相关 grant type
- callback handler

------

## 5）涉及哪些 rules

- **API2-R2**
- **API2-R5**
- **API2-R6**
- **API6-R1**

------

# 八、Device Code 和 User Code：设备类流程专用

这两个要放在一起讲，不然你会记不住。

------

## 1）Device Code 是什么

### 小白版定义

**Device Code** 是设备授权流程里的设备侧中间凭证。

### 通俗理解

设备先拿到 device code，然后在后台等用户完成授权。

------

## 2）User Code 是什么

### 小白版定义

**User Code** 是显示给用户输入的一段短码，用来把“这个设备”和“这个用户的授权动作”绑定起来。

### 通俗理解

你可以把它理解成：

> 设备显示一串码，用户去另一个更方便的设备上输入这串码，完成授权。

------

## 3）为什么这两个要区分

- **Device Code**：系统内部跟踪设备授权状态用
- **User Code**：给人看的、给人输入的

------

## 4）为什么 rules 会单独管它们

因为 device flow 有自己的风险点：

- code 生命周期
- user code 强度
- retry / failed attempts
- polling 频率
- 是否可存储

------

## 5）涉及哪些 rules

- **API2-R1**
- **API2-R2**
- **API2-R3**
- **API2-R5**

------

# 九、JWT Authentication Token / Client Assertion：客户端证明“我是我”的材料

这个概念比前面几个略高级，但你现在要先有直觉。

------

## 1）它是什么

### 小白版定义

这是客户端用来向授权服务器证明“我是某个合法 confidential client”的一种签名材料。

### 通俗理解

你可以把它理解成：

> “不是用户登录，而是应用自己出示一份签名证明，说‘我是这个 client，请给我 token’。”

------

## 2）它和 access token 的区别

这个一定要分开。

### Access Token

- 目的是去访问 API

### JWT Authentication Token / Assertion

- 目的是去向授权服务器证明客户端身份

也就是说：

- 一个是“调 API 用的”
- 一个是“申请 token 时证明 app 自己是谁用的”

------

## 3）你什么时候会看到它

你前面那段 Java `ApimOauth` 代码就是一个很好的例子：

- 从 JKS 里取私钥
- 构造 payload
- 生成 assertion / token
- 带着 `grant_type + assertion` 去 token endpoint
- 再换出 access token

这时前面生成的那个签名材料，就更接近 JWT auth token / client assertion 这种概念。

------

## 4）为什么企业喜欢这种方式

因为对于 confidential client 来说，它通常比简单密码更强：

- 私钥在服务端
- 可做签名验证
- 可配合证书 / JKS / mTLS
- 更适合 app-to-app 场景

------

## 5）涉及哪些 rules

- **API2-R1**
- **API2-R7**
- **API2-R3**
- **API2-R2**

------

# 十、Bearer：它不是 token 类型，而是 token 用法

这个你前面已经碰到了，现在正式系统讲一下。

------

## 1）它是什么

### 小白版定义

**Bearer** 指的是一种 token 的使用方式：
 谁持有这个 token，谁就可以拿它去访问资源。

### 通俗理解

你可以理解成：

> “持有者即有权使用。”

------

## 2）为什么这很重要

因为它揭示了一个风险本质：

> Bearer token 一旦泄露，别人拿着就能用。

系统一般不会再问：

- 这个 token 最初是谁拿到的

它只会问：

- token 是否有效
- 是否没过期
- scope 是否够
- signature 是否正确

------

## 3）它长什么样

最经典的就是：

```
Authorization: Bearer <access_token>
```

------

## 4）为什么 rules 会特别管 token 泄露

因为 Bearer 的特性决定了：

- 不要进 URL
- 不要进日志
- 不要乱存
- 生命周期要短
- 高敏感场景要考虑 sender-constrained 方案

------

## 5）涉及哪些 rules

- **API2-R5**
- **API2-R2**
- **API2-R7**
- **API6-R1**

------

# 十一、你做 assessment 时，怎么区分这些 token

这个特别实用。

以后不要一看到 token 就只写一句“项目使用 token”。

你应该先问：

------

## 1）这是拿来调 API 的，还是拿来表示登录结果的

- 调 API → Access Token
- 表示登录结果 → ID Token

------

## 2）这是长期续命的，还是短时访问的

- 续命 → Refresh Token
- 短时访问 → Access Token

------

## 3）这是用户场景的，还是 app 自己场景的

- 用户登录后拿的 token → user-based
- app 自己用 assertion 换来的 token → client/app-based

------

## 4）这是最终 token，还是中间凭证

- 中间凭证 → Authorization Code / Device Code / User Code
- 最终访问凭证 → Access Token

------

# 十二、你在项目里去哪里找这些 token

------

## 1）前端代码里

你可能会看到：

- login callback
- token storage
- `access_token`
- `id_token`
- refresh logic
- Bearer header injection

------

## 2）后端代码里

你可能会看到：

- token endpoint 调用
- `getAccessToken()`
- assertion generation
- JKS / keystore
- refresh / revoke 操作

------

## 3）网关配置里

你可能会看到：

- VerifyJWT
- OAuth policy
- scope check
- Authorization header handling

------

## 4）WebSSO / IdP 配置里

你可能会看到：

- token lifetimes
- allowed grant types
- refresh token policy
- client type
- redirect URIs

------

# 十三、这章和 rules 的对应关系

这章几乎和整个 Token 大类直接绑定。

------

## API2-R2 — Tokens lifespan

会直接问：

- access token 活多久
- ID token 活多久
- refresh token 活多久
- auth code 活多久
- device code 活多久

------

## API2-R3 — Tokens format

会问：

- access token 是 opaque 还是 JWT
- ID token 是否签名
- API key 是否够强
- user code 是否够复杂
- alg 是否安全

------

## API2-R5 — Tokens management

会问：

- token 存在哪
- 是否进日志
- 是否在 URL
- refresh token 是否 rotation
- logout 后是否撤销

------

## API2-R6

会问：

- Authorization Code 是否安全
- PKCE 是否启用

------

## API2-R7

会问：

- client auth token / assertion / mTLS 怎么做

------

## API6-R1

会问：

- 短时 token
- nonce
- one-time use
- replay 防护

------

# 十四、一个完整例子，把这章串起来

假设一个用户用 Angular + WebSSO 登录后访问账户 API。

### 第一步

用户被带到 WebSSO 登录。
 成功后，客户端拿到一个 **authorization code**。

### 第二步

客户端（或 BFF）拿 code 去换 token。
 可能拿到：

- **access token**
- **ID token**
- 有时还有 **refresh token**

### 第三步

前端/BFF 用 **access token** 去调 API。
 格式常常是：
 `Authorization: Bearer <access_token>`

### 第四步

前端可以根据 **ID token** 知道用户是谁、认证何时发生、认证方式是什么。

### 第五步

如果 access token 过期，有的系统会用 **refresh token** 去换新的 access token。

你看，一次正常流程里就可能同时出现 4 种不同 token/凭证。

------

# 十五、这一章最重要的 9 句话

## 第一句

**Token 不是一种东西，而是一整个家族。**

## 第二句

**Access Token 是给 API 用的。**

## 第三句

**ID Token 是给 Client 理解登录结果用的。**

## 第四句

**Refresh Token 是用来换新的 access token 的。**

## 第五句

**API Key 更像应用标识型凭证，不等于完整动态授权 token。**

## 第六句

**Authorization Code 是中间兑换券，不是最终 API 通行证。**

## 第七句

**JWT authentication token / assertion 更像客户端证明自己身份的材料，不是最终访问 API 的 token。**

## 第八句

**Bearer 不是 token 类型，而是“谁拿着谁能用”的使用方式。**

## 第九句

**做 assessment 时，先分清 token 是哪一种，再谈生命周期、格式、存储和规则合规。**

------

## 十六、一句话总结这章

你以后看到任何 token，都先问自己：
 **它是给谁用的、拿来干什么、处在流程的哪个阶段。**



# 第 7 部分：Token 里面的字段——claims 到底是什么

这一章你先抓住一句话：

> **Claim = token 里的一个字段。**

比如你看到一个 JWT 里有：

- `iss`
- `sub`
- `aud`
- `exp`

这些全都是 claims。

------

## 一、先说 claim 到底是什么

### 1）最简单定义

**Claim** 就是 token 里描述某些信息的字段。

### 2）通俗理解

你可以把 token 想成一张电子证件。
 那 claims 就像证件上写着的这些内容：

- 谁签发的
- 发给谁
- 代表谁
- 什么时候过期
- 能干什么

### 3）为什么重要

因为系统不是看“这串 token 长得像不像”，而是要看里面写了什么。

比如：

- 这个 token 是谁发的
- 是不是发给我这个 API 的
- 现在过期没
- 这个 token 代表的是用户还是客户端
- 认证强度够不够

这些都要通过 claims 来判断。

------

## 二、不是所有 token 都一定能直接看到 claims

这点先讲清楚，不然后面你会误会。

### 1）JWT 类型 token

如果 token 是 **JWT/JWS/JWE** 这种结构化 token，通常里面会有 claims。

### 2）Opaque token

如果 token 是 **opaque token**，那外面通常看不到 claims 内容。
 这时 claims 可能只存在于：

- Authorization Server
- Introspection endpoint
- API Gateway 内部解析结果

### 3）所以你现在先这样记

- **JWT**：你经常能直接看到 claims
- **Opaque token**：你可能看不到 claims，但系统内部仍然会基于某些信息判断 token

------

# 三、先建立最核心的 6 个 claims

你现在先优先记住这 6 个，最重要：

1. `iss`
2. `sub`
3. `aud`
4. `exp`
5. `nbf`
6. `kid`

后面再补几个扩展的。

------

# 四、iss（issuer）——谁签发了这个 token

------

## 1）是什么

**`iss` = issuer**，表示：

> 这个 token 是谁发出来的。

### 通俗理解

你可以把它理解成：

> “这张证是谁签发的？”

------

## 2）为什么重要

因为系统不能随便信任何 token。
 它必须确认：

- 这是不是我信任的那个身份系统发的
- 不是某个攻击者自己伪造的

### 银行/企业场景例子

如果你们公司规定只有公司 WebSSO / OpenID Provider 能发合法 token，
 那 API 或 Gateway 在校验 token 时就要检查：

- `iss` 是不是这个受信任的 issuer

------

## 3）在 assessment 里怎么用

你看到项目说“我们验证 token”，你要继续问：

- 有没有验证 `iss`
- 验证的是哪个 issuer
- 是不是和 WebSSO 配置一致

### 相关 rules

- **API2-R3**
- **API2-R7**
- **API0-R2**

------

# 五、sub（subject）——这个 token 代表谁

------

## 1）是什么

**`sub` = subject**，表示：

> 这个 token 的主体是谁。

### 通俗理解

你可以把它理解成：

> “这张证是发给谁/代表谁的？”

------

## 2）它可能代表什么

这个要看场景，不要死记。

### 场景 A：代表用户

那 `sub` 可能是：

- user id
- customer id
- employee id

### 场景 B：代表客户端

那 `sub` 可能是：

- client_id
- app identity

所以你不能看到 `sub` 就自动认为“这一定是用户”。

------

## 3）为什么重要

因为系统要知道：

- 这次请求到底是代表哪个 user
- 还是代表哪个 client

而很多授权、审计、追踪都依赖这个主体信息。

### 企业场景例子

在 **client authentication** 的 JWT 场景里，rule 甚至会要求：

- `sub` 要和 `client_id` 一致

------

## 4）在 assessment 里怎么用

你看到项目在做 JWT/client assertion 时，就要问：

- `sub` 放的是 user 还是 client
- 这个 `sub` 是否和场景匹配
- gateway / backend 有没有基于它做校验

### 相关 rules

- **API2-R7**
- **API1-R1**
- **API6-R1**

------

# 六、aud（audience）——这个 token 是给谁用的

------

## 1）是什么

**`aud` = audience**，表示：

> 这个 token 预期是发给谁使用的。

### 通俗理解

你可以把它理解成：

> “这张证是给哪个系统看的？”

------

## 2）为什么这个概念特别重要

因为 token 不是“哪个公司系统都能通用”。
 它应该有明确的目标对象。

### 例子

一个发给“账户 API”的 token，不应该拿去调用“贷款 API”或“别的资源服务器”。

------

## 3）为什么企业会特别强调 aud 校验

因为如果不校验 audience，就可能发生：

- 一个本来给系统 A 的 token
- 被错误拿到系统 B 去使用
- B 还错误接受了

这会扩大 token 的滥用范围。

------

## 4）在 assessment 里怎么用

你要问：

- API / Gateway 是否校验 `aud`
- `aud` 是否与当前 API / resource server 匹配
- client assertion 的 `aud` 是否对准 token endpoint / authorization server

### 相关 rules

- **API2-R7**
- **API2-R3**
- **API0-R2**

------

# 七、exp（expiration time）——什么时候过期

------

## 1）是什么

**`exp` = expiration time**，表示：

> 这个 token 到什么时间之后就不能再用了。

### 通俗理解

你可以把它理解成：

> “这张证几点之后作废。”

------

## 2）为什么特别重要

因为 token 一旦泄露，如果不过期或者活太久，攻击者就能长时间用它。

所以 access token、ID token、assertion 等都经常会有 `exp`。

------

## 3）在 assessment 里怎么用

你要看两件事：

### A. 有没有 `exp`

至少说明 token 不是无限期的。

### B. 过期时间是不是合理

这就要结合规则：

- access token 是否过长
- JWT auth token 是否应该更短
- authorization code 是否短时一次性

------

## 4）相关 rules

- **API2-R2**
- **API2-R7**
- **API6-R1**

------

# 八、nbf（not before）——在这个时间之前不能用

------

## 1）是什么

**`nbf` = not before**，表示：

> 这个 token 在某个时间点之前不应该被接受。

### 通俗理解

你可以把它理解成：

> “这张证现在还没到生效时间。”

------

## 2）为什么会需要它

有时 token 不是“立刻发出立刻就能用”，
 或者系统希望严格控制时间窗口。

### 企业意义

它能帮助系统更严格控制 token 的有效使用时间范围。

------

## 3）在 assessment 里怎么用

通常你会在以下场景看到它：

- JWT authentication token 校验
- 更严格的时间窗口控制
- rule 明确提到 API Provider 应校验 `exp` 和 `nbf`

### 相关 rules

- **API2-R7**
- **API6-R1**

------

# 九、kid（key ID）——这次签名用的是哪把 key

------

## 1）是什么

**`kid` = key identifier**，表示：

> 这份 token 的签名是用哪把 key 生成的。

### 通俗理解

你可以把它理解成：

> “这张证盖章时，用的是哪枚印章。”

------

## 2）为什么系统需要它

企业里经常不止一把签名 key。
 比如：

- 历史 key
- 新轮换的 key
- 不同 client 或不同 issuer 的 key

系统收到 token 后，需要知道：

- 去找哪把公钥来验签

------

## 3）为什么这对企业项目重要

因为 key rotation（密钥轮换）是常见操作。
 如果没有 `kid`，系统会更难管理多把 key。

------

## 4）在 assessment 里怎么用

看 client authentication / JWT validation 时可以问：

- token header 里是否带 `kid`
- 网关/API 是否根据 `kid` 找正确 key
- `kid` 是否和 issuer 关系一致

### 相关 rules

- **API2-R7**
- **API2-R3**

------

# 十、现在补 4 个在 OIDC 场景常见的 claims

这 4 个更多出现在 **ID token** 或认证上下文里。

1. `at_hash`
2. `acr`
3. `amr`
4. `auth_time`

------

# 十一、at_hash —— ID token 里和 access token 关联的哈希

------

## 1）是什么

**`at_hash`** 是 ID token 里关于 access token 的一个哈希值。

### 通俗理解

你可以把它理解成：

> “ID token 里放了一个 access token 的指纹，用来证明这两个是配套的一次登录结果。”

------

## 2）为什么重要

客户端可以用它来校验：

- 这次拿到的 ID token
- 和这次拿到的 access token
- 是不是同一轮认证流程出来的

------

## 3）在 assessment 里怎么用

当你看到 rule 提到 ID token 里要有 `at_hash`，
 你就知道它是在强调：

- client 不应随便把一个不相关 access token 和当前 ID token 混用

### 相关 rules

- **API2-R5**

------

# 十二、acr —— 认证强度等级

------

## 1）是什么

**`acr` = authentication context class reference**

### 小白版理解

它表示：

> 这次登录的认证强度/认证上下文级别是什么。

### 通俗理解

你可以理解成：

> “这次登录是普通级别，还是更强的安全级别。”

------

## 2）为什么重要

因为企业里不是所有 API 都要求同样强度的认证。

比如：

- 看公开信息可能弱一些
- 看账户数据、做转账可能要更强
- 高保密/高完整性 API 可能要求 MFA

`acr` 可以帮助客户端或系统知道：

- 这次认证强度够不够

------

## 3）和哪些 rules相关

尤其和：

- **API2-R9**（用户认证强度）
- **API2-R5**（ID token claims）
   有关系。

------

# 十三、amr —— 实际用了哪些认证方式

------

## 1）是什么

**`amr` = authentication methods reference**

### 通俗理解

它是在告诉你：

> 这次登录到底用了哪些认证方法。

### 例子

可能表示用了：

- password
- otp
- biometric

------

## 2）为什么重要

因为“强度够不够”不是只能靠抽象等级，有时也要看：

- 到底有没有用 MFA
- 是密码 + 短信
- 还是只有密码

------

## 3）在 assessment 里怎么用

如果你们项目需要高敏感认证，你就会关心：

- ID token / 身份上下文里能不能看出认证方式
- 是否支持用 `amr` 或类似信息判断

### 相关 rules

- **API2-R9**
- **API2-R5**

------

# 十四、auth_time —— 用户什么时候登录的

------

## 1）是什么

**`auth_time`** 表示：

> 用户这次认证是什么时间发生的。

### 通俗理解

你可以理解成：

> “这次登录是多久之前完成的。”

------

## 2）为什么重要

有些业务操作会关心：

- 这次登录是不是刚完成的
- 会不会太久之前登录过，现在还在沿用旧状态

### 企业场景例子

某些敏感动作前，系统可能希望：

- 如果登录太久之前了，要重新认证

------

## 3）和哪些 rules 相关

- **API2-R5**
- **API2-R8**
- **API2-R9**

------

# 十五、再讲两个你后面会常常碰到，但前面未展开的概念

------

## 1）scope claim

有些 token 里会有 scope 信息。

### 通俗理解

就是：

> 这张 token 被授予了哪些能力范围。

### 例子

- `read_accounts`
- `write_profile`

### 为什么重要

gateway 和 API 常常先用它做粗粒度授权。

### 相关 rules

- **API5-R1**
- **API1-R1**
- **API3-R4**

------

## 2）jti（如果你后面见到）

虽然你前面的 rules 没直接强调它，但你后面源码里可能会碰到。

**`jti` = JWT ID**

### 通俗理解

就是：

> 这张 token 自己的唯一编号。

### 为什么有用

可以帮助：

- 防重放
- 做 blacklist / revocation 跟踪

你现在先知道这个概念就行，不用展开记。

------

# 十六、你做 assessment 时，“校验 token”到底在校验什么

这是这一章最重要的落地点。

以后你看到别人说：

> “我们有校验 JWT / token validity”

你不要停在这里。
 你要继续问：

------

## 1）有没有校验 issuer（`iss`）

是不是受信任身份系统发的。

------

## 2）有没有校验 audience（`aud`）

是不是发给这个 API / 这个 token endpoint 用的。

------

## 3）有没有校验 subject（`sub`）

主体是不是符合预期。

------

## 4）有没有校验 expiration（`exp`）

有没有过期。

------

## 5）有没有校验 not-before（`nbf`）

是不是还没到可用时间。

------

## 6）有没有校验 signature

这个虽然不是 claim，但和 claim 一起构成完整校验。
 也就是：

- 不是被篡改的
- 确实是受信 issuer 用正确 key 签的

------

## 7）对于 OIDC，有没有处理 `at_hash` / `acr` / `amr` / `auth_time`

特别是涉及：

- 用户身份
- MFA
- 敏感操作
- 登录上下文

时，这些 claim 才真正体现出价值。

------

# 十七、你在项目里去哪里看这些 claims

------

## 1）在 JWT 本身里看

如果 token 是 JWT，你经常可以：

- 解码 header/payload
- 看到里面的 claims

注意：

- 解码能看，不代表你就信
- 真正安全性还要看签名校验

------

## 2）在网关配置里看

比如 Apigee / gateway policy 可能会配置：

- 验 `iss`
- 验 `aud`
- 验 `exp`
- 验签名

------

## 3）在后端代码里看

后端常见会：

- parse JWT
- 读取 `sub`
- 读取 `scope`
- 读取 `acr/amr`
- 做 `@PreAuthorize` 或业务校验

------

## 4）在客户端/BFF里看

尤其 OIDC 场景下，客户端可能会读取：

- `id_token`
- `auth_time`
- `amr`
- `acr`
- `at_hash`

------

# 十八、这一章和 rules 的对应关系

------

## API2-R3 — Tokens format

这一条最直接相关，因为它要求：

- token 格式要安全
- JWT 要符合标准
- alg 不能是 `none`

而 claims 正是 JWT 格式里最核心的内容之一。

------

## API2-R7 — Client authentication

它会非常关注：

- `iss`
- `sub`
- `aud`
- `exp`
- `nbf`
- `kid`

因为这些是 client assertion / JWT auth 校验的核心。

------

## API2-R5 — Tokens management

它会涉及：

- `at_hash`
- `acr`
- `amr`
- `auth_time`
- ID token 的用途边界

------

## API2-R9 — End-user authentication

它会特别和：

- `acr`
- `amr`
- `auth_time`

这些认证上下文 claims 有关系。

------

## API6-R1 — Replay attacks

它会和：

- `exp`
- `nbf`
- `nonce`
- 一次性使用

这些一起形成防重放控制。

------

# 十九、一个完整例子，把这章串起来

假设一个用户登录后，客户端拿到一个 ID token 和一个 access token。

### 你做 assessment 时可能要问：

#### 1. `iss`

是不是你们公司的 WebSSO 发的？

#### 2. `aud`

是不是发给当前这个 client / API 的？

#### 3. `sub`

代表的是哪个用户？

#### 4. `exp`

有没有过期？

#### 5. `acr` / `amr`

这次是不是 MFA？认证方式够不够？

#### 6. `auth_time`

是不是太久之前认证的？

#### 7. `at_hash`

ID token 和当前 access token 是不是一套配对结果？

你看，“校验 token”一下子就不再是空话了，而是变成一组具体检查点。

------

# 二十、这一章最重要的 9 句话

## 第一句

**Claim 就是 token 里的字段。**

## 第二句

**系统是否信任 token，不是看它长得像不像，而是看 claims 和签名是否都符合预期。**

## 第三句

**`iss` 解决“谁发的”，`sub` 解决“代表谁”，`aud` 解决“给谁用”。**

## 第四句

**`exp` 和 `nbf` 解决时间有效性问题。**

## 第五句

**`kid` 帮系统找到正确的 key 去验签。**

## 第六句

**`acr`、`amr`、`auth_time` 更偏认证上下文，而不是普通 API 调用权限。**

## 第七句

**`at_hash` 帮客户端把 ID token 和 access token 关联起来。**

## 第八句

**做 assessment 时，“验证 token”要落到具体 claim 级别。**

## 第九句

**不是所有 token 都能直接看到 claims，但 claims 思维仍然重要。**

------

## 二十一、一句话总结这章

你以后看到任何 JWT / token 校验，都先问自己：
 **它到底校验了谁发的、代表谁、给谁用、什么时候过期，以及认证上下文够不够。**



# 第 8 部分：各种 ID，不要再混

你先抓住一句话：

> **不同的 ID，回答的是不同的问题。**

有的 ID 在回答：

- 这是哪个应用？

有的在回答：

- 这是哪个用户？

有的在回答：

- 这是哪个业务对象？

有的在回答：

- 这是哪一次请求链路？

所以你以后看到某个 `id`，第一反应不要是“这是一个编号”，而要问：

> **它到底在标识什么。**

------

## 一、先给你一张总图

你当前最需要掌握的几类 ID 是：

1. **client_id**
2. **user id / user identity**
3. **subject id（通常体现在 `sub`）**
4. **object id**
5. **API key（有时也承担调用方标识）**
6. **correlation id**
7. **key id（`kid`）**

你可以先把它们分成 4 组：

### 第 1 组：谁在发请求

- client_id
- API key（部分场景）
- client assertion 里的 subject

### 第 2 组：最终是谁在使用系统

- user id
- subject（用户场景下）

### 第 3 组：请求访问的是哪个业务对象

- accountId
- customerId
- contractId
- orderId

### 第 4 组：怎么追踪一次技术调用

- correlation id
- trace id
- key id（更偏密钥识别）

------

# 二、client_id：客户端应用的身份编号

------

## 1）它是什么

### 小白版定义

**client_id** 是授权服务器 / WebSSO / OAuth 系统分配给某个客户端应用的身份编号。

### 通俗理解

你可以把它理解成：

> “这是哪个应用在接入我。”

不是“哪个用户”，而是“哪个应用”。

------

## 2）最典型的例子

假设你们有：

- Angular 前端
- Mobile App
- BFF
- 内部批处理程序

它们理论上都应该有自己独立的 `client_id`。

所以可能会是：

- `customer-portal-web`
- `customer-mobile-app`
- `kat-bff`
- `batch-reporting-service`

实际值可能不是这么可读，但意思一样。

------

## 3）它不是什么

### 它不是用户 ID

`client_id` 不表示张三、李四是谁。
 它表示的是：

- Angular app
- 某个 BFF
- 某个 backend service

### 它也不是业务对象 ID

它不表示 account、order、customer 这些业务数据对象。

------

## 4）为什么它重要

因为很多安全控制都是围绕 **哪个 client 在接入** 来做的：

- 允许哪些 grant flow
- 允许哪些 scope
- 是 public client 还是 confidential client
- 是否要求 mTLS
- 是否要求 JWT client authentication
- rate limit / quota 按谁计算
- 审计时是谁调用了 API

------

## 5）在代码和配置里会出现在哪里

你以后常见位置有：

### 前端 / 登录配置

- OIDC client config
- login config
- environment 配置

### 后端

- `@Value("${...clientid}")`
- token 请求参数
- assertion payload
- WebSSO client registration

### 网关和平台

- API product / app registration
- allowed client list
- token validation context

------

## 6）在 assessment 里怎么用

看到 `client_id` 时，你要问这些问题：

### A. 这个 client_id 对应的是哪个组件

它对应的是：

- Angular 前端？
- BFF？
- mobile app？
- 后端服务？

### B. 是否唯一

是不是多个系统共用了同一个 client_id。

### C. 是否和它的能力匹配

比如：

- public client 却拿了服务端才能用的能力
- 某个只读 app 却拿到了高权限 scope

### D. 是否和证书/subject 绑定

在更强的客户端认证场景里，client_id 还应该和证书 subject 或 JWT subject 有对应关系。

------

## 7）主要涉及哪些 rules

- **API2-R10**：直接要求 client identifier 唯一
- **API2-R1**：不同 client 类型决定不同 flow
- **API2-R7**：client authentication 时常要求和 `sub` / certificate subject 对应
- **API5-R1**：scope 分配和 client 绑定
- **API4-R1**：rate limit 常按 client_id 做
- **API9-R1**：治理注册表里要记录 authorized client applications

------

# 三、user id / user identity：最终用户是谁

------

## 1）它是什么

### 小白版定义

**user id** 是最终用户的身份标识。

### 通俗理解

你可以把它理解成：

> “到底是哪一个人正在使用这个系统。”

------

## 2）常见例子

- 客户编号
- 员工工号
- 登录账号标识
- 某个身份中心里的唯一用户标识

------

## 3）为什么它重要

因为很多业务授权最终看的是：

- 这个 user 能不能访问这个资源
- 这个 user 的权限等级是什么
- 这个 user 是否通过了足够强的认证

不是只看：

- 哪个 app 在调

------

## 4）在 token / 系统里可能怎么出现

它可能出现在：

- `sub`
- userinfo
- session context
- security context
- backend identity propagation header

------

## 5）在 assessment 里怎么用

看到用户身份时，你要问：

### A. 后端最终知道是哪个 user 吗

有些系统只有 client 身份，没有 user 身份，这会导致细粒度授权做不起来。

### B. user identity 有没有在 gateway 到 backend 之间正确传递

如果 gateway 验了 token，但 backend 不知道最终用户是谁，也会有问题。

### C. user identity 是否被错误信任

比如前端自己传一个 `userId=123`，后端就信了，这就是危险信号。

------

## 6）主要涉及哪些 rules

- **API1-R1**
- **API3-R1**
- **API3-R4**
- **API2-R8**
- **API2-R9**

------

# 四、subject（`sub`）到底和 user id 是什么关系

这个地方很容易混，所以要单独拆开。

------

## 1）`sub` 是什么

前面你学过，`sub` 是 token 里的 **subject claim**，表示：

> 这张 token 代表的主体是谁。

------

## 2）为什么它不等于“永远是 user id”

因为 `sub` 取决于场景。

### 场景 A：用户登录型 token

那 `sub` 很可能就是 user identity。

### 场景 B：客户端认证型 JWT

那 `sub` 可能是 client_id。

------

## 3）所以你以后不要这样想

不要看到 `sub` 就自动说：

> “这是用户 ID。”

正确说法应该是：

> “这是 token 的主体标识，具体是 user 还是 client，要结合场景看。”

------

## 4）在 assessment 里怎么用

### 在 user-based flow 里

你会关心：

- `sub` 是否对应最终用户

### 在 client authentication 里

你会关心：

- `sub` 是否与 client_id 一致

------

## 5）主要涉及哪些 rules

- **API2-R7**
- **API1-R1**
- **API0-R2**

------

# 五、object id：业务对象的编号

这个是你后面做越权判断时最常见、也最关键的一类 ID。

------

## 1）它是什么

### 小白版定义

**object id** 是业务数据对象的编号。

### 通俗理解

你可以把它理解成：

> “请求正在访问哪一个具体业务对象。”

------

## 2）常见例子

- `accountId`
- `customerId`
- `orderId`
- `contractId`
- `loanId`
- `invoiceId`

------

## 3）为什么它重要

因为大量 API 越权问题，根本不是 token 校验错了，而是：

- 用户已登录
- app 也合法
- 但 object id 被改了
- 后端没重新校验 ownership / authorization

这就是你前面接触到的：

- BOLA
- IDOR

本质上都和 object id 管理有关。

------

## 4）一个最经典的危险例子

用户原本请求：

```
GET /accounts/123
```

然后把路径改成：

```
GET /accounts/124
```

如果系统只验证“你登录了”，没验证“124 是不是你的账户”，那就越权了。

这里真正出问题的不是 authentication，而是：

> **object-level authorization 没跟 object id 做绑定检查。**

------

## 5）在 assessment 里怎么用

看到 object id 时，你要问：

### A. 这个 ID 是用户可控的吗

- path parameter
- query parameter
- request body field

如果用户能改，就要特别警惕。

### B. 后端有没有重新校验它和当前 user 的关系

比如：

- 这个 accountId 是不是当前用户的
- 这个 contractId 是否属于当前客户
- 这个 orderId 是否属于当前 tenant

### C. 是否只是“查得到对象就返回”

如果后端只是 `findById(id)` 然后直接返回，那风险很大。

------

## 6）主要涉及哪些 rules

- **API1-R1**
- **API3-R1**
- **API3-R4**
- **API7-R4**
- 和你前面 Apigee 文档里的 **BOLA / IDOR** 强相关

------

# 六、API key：有时候也在“标识调用方”

这个前面讲过 token 家族，这里从 ID 角度再补一下。

------

## 1）为什么这里还要再提 API key

因为在某些场景下，API key 不只是“一个密钥”，它也承担：

> “这个调用方是谁”的基础识别作用。

所以从治理和审计角度，它有时近似承担 client identifier 的角色。

------

## 2）它和 client_id 的区别

### client_id

更偏 OAuth/OIDC 体系里的客户端身份标识。

### API key

更偏简单接入标识 / 流量归属标识。

------

## 3）为什么不能把它们完全等同

因为 API key 通常没有 OAuth 体系里那么完整的上下文：

- 不一定有 user 维度
- 不一定有标准 scope
- 不一定有强认证能力
- 不一定适合高敏感授权

------

## 4）在 assessment 里怎么用

如果某项目主要靠 API key 接入，你要特别问：

- 这是公开数据场景吗
- 它是不是被误当成强认证机制
- 是不是还缺更强的 client authentication

------

## 5）涉及哪些 rules

- **API2-R2**
- **API2-R3**
- **API2-R5**
- **API2-R10**

------

# 七、correlation id：不是身份 ID，而是“请求追踪 ID”

这类 ID 很容易被忽视，但做企业 assessment 时非常重要。

------

## 1）它是什么

### 小白版定义

**correlation id** 是一笔请求在多个系统之间流转时的追踪编号。

### 通俗理解

你可以把它理解成：

> “这一整串调用链的工单号。”

------

## 2）为什么要有它

因为企业系统往往不是一个服务就结束，而是：

- 前端
- gateway
- BFF
- backend service A
- backend service B
- DB / downstream

如果没有一个统一编号，你很难追踪：

- 某次异常是哪条请求链触发的
- 哪一步失败
- 哪一步做了越权尝试
- 某个用户的这次调用穿过了哪些系统

------

## 3）它和 user id / client_id 的区别

### user id

回答：是谁在用系统

### client_id

回答：哪个应用在调用

### correlation id

回答：这是哪一次调用链路

它不回答“身份”，它回答“轨迹”。

------

## 4）在项目里长什么样

可能会出现在：

- HTTP header
- gateway logs
- backend logs
- distributed tracing
- SIEM / monitoring 平台

常见命名像：

- `X-Correlation-ID`
- `traceId`
- `requestId`

------

## 5）在 assessment 里怎么用

你看到这类字段时，要问：

- 是否贯穿 gateway 到 backend
- 是否会在日志里统一记录
- 是否足以追溯异常和安全事件

------

## 6）主要涉及哪些 rules

- **API10-R1**（traceability）
- **API7-R6**
- **API7-R7**（测试与追踪时也常用）

------

# 八、key id（`kid`）：不是业务身份，而是“签名密钥的编号”

这个你前面在 claims 那章已经见过，这里从“各种 id 不要混”的角度再定一下。

------

## 1）它是什么

**`kid`** 是：

> 哪把签名 key 被用来给 token 签名的标识。

------

## 2）它不是什么

它不是：

- user id
- client_id
- object id
- correlation id

------

## 3）为什么重要

因为企业里常常有多把 key：

- 历史 key
- 新轮换的 key
- 多个 issuer 的 key

系统得知道该用哪把公钥去验签。

------

## 4）在 assessment 里怎么用

你不会拿它做业务授权，但你会在 token validation / client authentication 里问：

- 是否有 `kid`
- gateway / API 是否按 `kid` 找 key
- key rotation 是否可管理

------

## 5）主要涉及哪些 rules

- **API2-R7**
- **API2-R3**

------

# 九、把这些 ID 放回真实请求里，你就不容易混了

我们来做一个完整例子。

假设一个用户通过 Angular 调一个账户接口。

### 用户是张三

这对应的是：

- **user id**

### Angular 前端是一个已注册客户端

这对应的是：

- **client_id**

### token 里的 `sub`

这可能表示：

- 如果是用户 access token，可能是张三这个 user
- 如果是 client assertion，可能是前端/BFF 的 client identity

### 请求访问的是账户 123

这对应的是：

- **object id = accountId 123**

### 这一整次调用链，从前端到 gateway 到 backend

这应该有一个：

- **correlation id**

### token header 里有 `kid`

这表示：

- 这张 token 是哪把 key 签的

你看，这 6 个 “id” 同时都可能出现，但它们完全不是一回事。

------

# 十、做 assessment 时，这一章怎么真正用起来

以后你一看到 “id”，固定先问下面这些问题。

------

## 1）这个 id 标识的是“应用”、"人"、"对象"还是“请求链路”？

这是第一判断。

------

## 2）如果是 client_id

问：

- 是否唯一
- 对应哪个组件
- 是否和 flow / scope / client type 一致

------

## 3）如果是 user identity

问：

- backend 是否真的知道最终 user 是谁
- 是否被错误伪造
- 是否贯穿到细粒度授权逻辑

------

## 4）如果是 object id

问：

- 用户是否可控
- 后端是否重新校验 ownership / authorization
- 是否存在 BOLA / IDOR 风险

------

## 5）如果是 correlation id

问：

- 是否贯穿日志与链路
- 是否足以支撑 traceability

------

## 6）如果是 `kid`

问：

- token validation / key rotation 是否完善

------

# 十一、这一章和 rules 的对应关系

------

## client_id 相关

- **API2-R10**
- **API2-R1**
- **API2-R7**
- **API5-R1**
- **API9-R1**

------

## user identity 相关

- **API1-R1**
- **API3-R1**
- **API3-R4**
- **API2-R9**

------

## object id 相关

- **API1-R1**
- **API3-R1**
- **API7-R4**
- 以及 BOLA / IDOR 类风险分析

------

## correlation id / traceability 相关

- **API10-R1**
- **API7-R6**
- **API7-R7**

------

## `kid` 相关

- **API2-R7**
- **API2-R3**

------

# 十二、这一章最重要的 8 句话

## 第一句

**不是所有 `id` 都是“用户编号”。**

## 第二句

**client_id 标识的是应用，不是人。**

## 第三句

**user id 标识的是最终用户身份。**

## 第四句

**object id 标识的是业务对象，是越权检查的关键。**

## 第五句

**correlation id 标识的是一次请求链路，不是身份。**

## 第六句

**`sub` 是 token 主体标识，但它可能代表 user，也可能代表 client。**

## 第七句

**`kid` 是签名密钥编号，不是业务 ID。**

## 第八句

**做 assessment 时，先分清 `id` 在标识什么，再判断它是否被正确使用。**

------

## 十三、一句话总结这章

你以后看到任何 `id`，先不要急着理解成“编号”，而要先问：
 **它到底在标识应用、用户、对象、请求链路，还是签名密钥。**



# 第 9 部分：Token 安全——生命周期、格式、存储、撤销

这一章你先抓住一句话：

> **Token 安全不是“有没有 token”，而是“这个 token 能活多久、能被谁看到、被偷了之后还能不能继续用”。**

------

## 一、先建立总图：token 安全看哪 4 个维度

以后你做 assessment，只要看到 token，就固定从这 4 个维度往下问：

### 1. 生命周期（Lifespan）

- 这个 token 活多久
- 是不是太长
- 过期后会怎样

### 2. 格式（Format）

- 是 opaque 还是 JWT
- 能不能被客户端读懂
- 签名/加密是否合理

### 3. 存储与传输（Storage & Transport）

- 存在哪里
- 会不会进日志 / URL / 浏览器存储
- 是否以正确方式传输

### 4. 撤销与轮转（Revocation & Rotation）

- 失效后能不能立即停用
- refresh token 是否轮转
- logout 后会不会真正作废

你可以把这 4 个维度想成 assessment 里的一个固定模板。

------

# 二、生命周期（Lifespan）：token 活多久

------

## 1）为什么生命周期这么重要

因为 token 一旦泄露，攻击者最开心看到的就是：

> **它还能用很久。**

如果 token 非常短命，攻击者可利用的窗口就小很多。
 如果 token 活得很久，风险窗口就会被拉大。

所以企业规则会强调：

- access token 要短命
- authorization code 要更短命
- refresh token 要谨慎
- 高风险 token 更要短命

------

## 2）先分清“不同 token，寿命不一样”

这点特别重要。

### Access Token

通常应该比较短。

### ID Token

一般也不应该特别长，很多时候会和 access token 接近。

### Refresh Token

通常比 access token 长，但因此风险也更高。

### Authorization Code

应该非常短，而且一次性使用。

### JWT Authentication Token / Assertion

通常也应该很短，因为它是拿来证明 client 身份的，不该长期重放。

------

## 3）为什么 access token 要短

因为它是直接调 API 的通行证。

### 如果它太长

一旦泄露：

- 别人直接可以拿去调 API
- 很难第一时间发现
- 风险会持续很久

所以 rule 才会倾向要求：

- access token 短时有效
- refresh token 谨慎使用
- 敏感场景不要让 token 长时间存活

------

## 4）为什么 authorization code 要更短

因为它虽然不是最终 access token，但它是“中间兑换券”。

如果 authorization code 被截获，攻击者可能去换 token。
 所以它应该：

- 很快过期
- 只能一次性使用
- 最好再配 PKCE

------

## 5）为什么 refresh token 风险更大

因为 refresh token 的价值在于“续命”。

也就是说：

- access token 过期了
- refresh token 还能换新的

这意味着：
 如果 refresh token 被偷，它可能带来持续访问能力，而不只是一次短时攻击窗口。

这也是为什么 browser-based public client 场景里，企业规则会对 refresh token 特别敏感。

------

## 6）在 assessment 里怎么检查生命周期

你要看：

### A. 配置值

比如：

- token TTL
- refresh token validity
- code lifetime

### B. WebSSO / IdP 设置

比如：

- access token 有效期
- ID token 有效期
- refresh token policy

### C. 代码逻辑

比如：

- 生成 assertion 时 `exp`
- 是否有 `issueTime + duration`
- refresh 行为怎么触发

### D. 实际响应

比如 token response 里是否有：

- `expires_in`

------

## 7）主要涉及哪些 rules

- **API2-R2**：最核心
- **API2-R4**：和 SSO lifespan 协调
- **API6-R1**：短生命周期帮助防 replay
- **API2-R8**：会话与 token 生命周期关系

------

# 三、格式（Format）：token 长什么样，为什么有的能看懂，有的看不懂

------

## 1）为什么 token 格式也会影响安全

因为不同格式意味着不同暴露面。

### 如果 token 是 opaque

客户端通常看不懂内容。

### 如果 token 是 JWT

客户端往往能解码 payload，看到 claims。

这并不自动说明哪种一定更安全，而是要看：

- 用在什么场景
- 谁能看到
- 是否签名
- 是否加密

------

## 2）Opaque Token

### 小白版定义

一串外部看不懂的随机字符串。

### 通俗理解

像一张“黑盒门票”。

### 特点

- 客户端通常不知道里面写了什么
- 适合不希望前端读内容的场景
- 很多企业会偏好 public client 使用这种方式

### 风险思维

虽然外部看不懂，但：

- 如果泄露，仍然可能被拿去用
- 所以生命周期和传输仍然重要

------

## 3）JWT

### 小白版定义

一种结构化 token，里面可以带 claims。

### 通俗理解

像一张“有字段内容的电子证件”。

### 特点

- 可以解码看到 payload
- 适合携带 issuer、subject、audience、expiry、scope 等信息
- 常见于 access token、ID token、client assertion

### 风险思维

JWT “能解码” 不等于 “不安全”。
 真正关键是：

- 有没有签名
- 有没有加密
- 用在什么场景
- 客户端是否不该看到里面内容

------

## 4）JWS

### 小白版定义

签名版 JWT。

### 通俗理解

内容可看，但带防篡改签名。

### 重点

它主要保证：

- 完整性
- 来源可信（前提是验证签名）

它**不保证保密性**，因为 payload 经常可以被解码看到。

------

## 5）JWE

### 小白版定义

加密版 JWT。

### 通俗理解

不仅有结构，而且内容被加密。

### 重点

它更适合：

- 不希望 token 内容被 public client 看到的场景
- 对保密性要求更高的场景

------

## 6）为什么 rules 会关心格式

因为企业规则在问：

- public client 能不能读懂 access token
- token 是否够复杂、够随机
- JWT 是否符合标准
- 有没有禁用 `alg=none`
- 是否应该优先 opaque
- 是否该用 JWE

------

## 7）在 assessment 里怎么查格式

### A. 看 token 长相

如果像三段式：

- `xxx.yyy.zzz`
   通常像 JWT

如果像一串随机字符串：

- 可能更像 opaque

### B. 看 IdP / gateway / app 配置

确认：

- access token format
- ID token format
- 是否加密
- 使用的签名算法

### C. 看代码或文档

比如：

- VerifyJWT
- JWT decoder
- introspection
- JWK set

------

## 8）主要涉及哪些 rules

- **API2-R3**
- **API2-R7**
- **API2-R5**

------

# 四、存储（Storage）：token 放在哪儿

这部分是最容易出事故的地方之一。

------

## 1）为什么存储位置这么关键

因为很多攻击不是“破解 token”，而是：

> **直接拿到已经发给你的 token。**

而 token 最容易泄露的地方，往往就是它的存储点。

------

## 2）浏览器里的存储风险

### A. LocalStorage / SessionStorage

这是你在前端项目里最该警惕的地方之一。

#### 为什么敏感

如果前端被 XSS 打穿，攻击者很容易把这些内容读出来。

#### assessment 时要问

- access token 是否放在 localStorage
- refresh token 是否放在浏览器可持久读取位置

### B. 内存（memory）

对 access token 来说，这通常比持久化存储更安全一些。

#### 为什么

因为刷新页面、重启浏览器后会丢失，攻击面相对小些。

### C. Cookie

Cookie 既可能是好事，也可能带来新问题。

#### 好处

- 某些 cookie 可以更受控
- 在 BFF 场景里浏览器不必直接持有业务 token

#### 风险

- 如果是 cookie session 模式，要考虑 CSRF
- cookie 也要正确设置 secure / httpOnly / sameSite 等属性（这是扩展知识，你后面会慢慢接触）

------

## 3）服务端里的存储风险

### 场景

- BFF
- Spring 后端
- batch 服务
- confidential client

### 要问什么

- token / refresh token 是否加密存储
- 配置是否在 Vault / secret 管理系统里
- 私钥是否在 JKS / keystore / HSM / Vault 中
- 会不会打印到日志

------

## 4）移动端里的存储风险

如果是 native mobile app，企业通常会更关注：

- secure element
- biometrics
- OS 安全容器

这也是为什么 rules 对 mobile refresh token 场景会写得比浏览器宽一些，但依然要求更安全存储。

------

## 5）在 assessment 里怎么检查存储

### 前端

看：

- localStorage/sessionStorage 使用
- auth service
- interceptor
- token service

### 后端

看：

- 配置文件
- secret 注入方式
- 是否落库
- 是否加密
- 是否打日志

### 平台

看：

- K8S Secret
- Vault
- JKS / SSL bundle
- client credentials 来源

------

## 6）主要涉及哪些 rules

- **API2-R5**
- **API0-R2**
- **API2-R8**

------

# 五、传输（Transport）：token 是怎么在请求里带出去的

这和存储不同，但紧密相关。

------

## 1）最推荐的方式

通常是通过 HTTP header 传输，最常见：

```
Authorization: Bearer <access_token>
```

### 为什么这样更好

因为相对 URL 参数来说，它不那么容易：

- 出现在浏览器历史里
- 出现在 referer 里
- 出现在日志里
- 出现在代理缓存记录里

------

## 2）为什么不该把 token 放 URL 里

如果 token 放在 query parameter，比如：

```
GET /api/accounts?access_token=...
```

风险会明显增大，因为它可能出现在：

- 浏览器历史
- 代理日志
- referrer header
- 监控日志
- 第三方分析工具

这就是为什么 rules 明确会说：

- token 不得通过 URL query parameter 传

------

## 3）在 assessment 里怎么检查传输方式

看：

- 前端请求代码
- RestTemplate / WebClient 调用
- 网关调用日志
- Swagger 示例
- 是否有 query parameter 传 token

------

## 4）主要涉及哪些 rules

- **API2-R5**
- **API2-R7**
- **API0-R1**

------

# 六、撤销（Revocation）：token 能不能被提前作废

------

## 1）为什么需要撤销

因为“过期”太慢了。

如果用户：

- logout
- 修改密码
- 撤销 consent
- 发现账号异常
- client 被停用

这时不能等 token 自然过期，而应该尽快失效。

------

## 2）通俗理解

撤销就是：

> **把原本还没过期的 token 提前作废。**

------

## 3）哪些场景特别依赖撤销

- logout
- refresh token 泄露
- 安全事件
- 用户权限变化
- consent 变化
- 客户端停用

------

## 4）为什么 access token 和 refresh token 的撤销关注点不同

### Access Token

通常较短命，所以有些系统更多依赖“等它快过期”。

### Refresh Token

通常更长命，所以更需要：

- 主动撤销
- rotation
- 异常检测

------

## 5）在 assessment 里怎么查撤销

看：

- WebSSO / token endpoint 是否支持 revoke
- logout 是否触发 revoke
- refresh token 失效策略
- 被撤销后再用是否会失败

------

## 6）主要涉及哪些 rules

- **API2-R5**
- **API2-R8**
- **API2-R2**

------

# 七、轮转（Rotation）：尤其是 refresh token rotation

------

## 1）什么叫轮转

最典型就是：

> 每次用 refresh token 换新 access token 时，同时把旧 refresh token 作废，并发一个新的 refresh token。

------

## 2）为什么要这样做

因为如果 refresh token 被偷，攻击者和合法用户都想用它时，系统就可以更早发现异常。

### 不轮转会怎样

- 一个长期 refresh token 被偷后
- 攻击者可以长期偷偷续命

### 轮转后会怎样

- 每次都换新
- 旧的不能再用
- 更容易识别“同一个旧 token 被重复使用”的异常

------

## 3）这为什么对浏览器尤其敏感

因为浏览器本来就不适合长期保敏感凭证。
 所以一旦企业允许某些 refresh token 场景，就会很强调 rotation。

------

## 4）在 assessment 里怎么查

看：

- token refresh 逻辑
- IdP refresh token policy
- 是否 one-time use
- 旧 refresh token 是否立即失效

------

## 5）主要涉及哪些 rules

- **API2-R5**
- **API2-R2**

------

# 八、Token 泄露（Leakage）：泄露点在哪里

这一部分非常适合 assessment 思维。

------

## 1）最常见泄露点

- 浏览器 localStorage
- URL query parameter
- referer
- 应用日志
- 网关日志
- 异常堆栈
- 前端 console
- 截图/调试工具
- 第三方脚本

------

## 2）为什么 rules 一直强调不要进日志和 URL

因为现实世界里，很多 token 泄露不是被“破解”，而是被“记录”了。

------

## 3）在 assessment 里怎么问

看到任何 token 逻辑，都要问：

- 是否可能出现在 URL
- 是否可能被 debug log 打印
- 是否可能出现在 frontend error log
- 是否被某些中间件默认记录

------

## 4）主要涉及哪些 rules

- **API2-R5**
- **API2-R2**
- **API10-R1**

------

# 九、把这 4 个维度放回不同 token 上，你就会更清楚

现在把不同 token 放到同一个框架里看。

------

## 1）Access Token

### 你主要关心

- 生命周期要短
- 传输要安全（header，不是 URL）
- 存储要谨慎
- 泄露风险高
- 通常是 Bearer

### 相关 rules

- API2-R2
- API2-R3
- API2-R5
- API6-R1

------

## 2）ID Token

### 你主要关心

- 用途要正确（给 client，不是给 API）
- 不应到处广播
- claims 是否符合标准
- 生命周期不应过长

### 相关 rules

- API2-R2
- API2-R3
- API2-R5

------

## 3）Refresh Token

### 你主要关心

- 风险更高
- 存储要求更严格
- rotation 是否启用
- 是否可以撤销
- 浏览器是否不该持有它

### 相关 rules

- API2-R2
- API2-R5
- API2-R8

------

## 4）Authorization Code

### 你主要关心

- 生命周期很短
- 一次性使用
- 和 client 绑定
- 是否配 PKCE

### 相关 rules

- API2-R2
- API2-R5
- API2-R6
- API6-R1

------

# 十、你做 assessment 时，这一章怎么落地成问题清单

以后看到项目里的 token，不要只是说“项目有 token 管理”。
 你应该固定问这 8 个问题：

### 1. 这是什么 token

access / ID / refresh / code / assertion？

### 2. 它活多久

TTL 是否合理？

### 3. 它是什么格式

opaque / JWT / JWS / JWE？

### 4. 它存在哪里

浏览器内存、cookie、localStorage、服务端、JKS、Vault？

### 5. 它怎么传

Authorization header、cookie，还是 URL？

### 6. 它会不会被记录

日志、错误堆栈、前端调试、监控系统？

### 7. 它能不能被撤销

logout、安全事件、权限变化时是否能提前失效？

### 8. 如果是 refresh token，是否轮转

one-time use 还是长期固定不变？

------

# 十一、这一章和 rules 的对应关系

------

## API2-R2 — Tokens lifespan

最直接，重点看：

- 各种 token 活多久

------

## API2-R3 — Tokens format

重点看：

- opaque / JWT / JWE
- 签名算法
- 随机性和复杂度

------

## API2-R5 — Tokens management

重点看：

- 存在哪
- 怎么传
- 是否 rotation
- 是否 revoke
- 是否进日志

------

## API2-R8 — Session management

重点看：

- logout 后 token 怎么处理
- token 和会话失效如何协调

------

## API6-R1 — Replay attacks

重点看：

- token 是否够短命
- code 是否一次性
- refresh token 是否轮转

------

# 十二、一个完整例子，把这章串起来

假设一个 BFF 去 WebSSO 拿 token 再调用 Apigee。

### 你应该怎么分析

#### 1. Token 类型

BFF 拿到的是 access token，可能还有 refresh token。

#### 2. 生命周期

看 access token TTL 是否合理，refresh token 是否过长。

#### 3. 格式

看 access token 是 opaque 还是 JWT。

#### 4. 存储

因为 BFF 是服务端，token 不该进浏览器；应该只在服务端管理。

#### 5. 传输

BFF 调 Apigee 时应通过 `Authorization: Bearer ...`。

#### 6. 泄露

检查日志里有没有打印 token。

#### 7. 撤销

logout / 异常时是否 revoke。

#### 8. 轮转

如果用了 refresh token，是否 one-time rotation。

你看，这一整套问题都来自这一章。

------

# 十三、这一章最重要的 10 句话

## 第一句

**Token 安全不在于“有没有 token”，而在于“它怎么被管理”。**

## 第二句

**生命周期越长，泄露后的风险窗口越大。**

## 第三句

**Access token 应短命，refresh token 风险更高。**

## 第四句

**Authorization code 不是最终 token，但也必须短时且一次性。**

## 第五句

**Opaque token 和 JWT 的暴露面不同，不能混着理解。**

## 第六句

**Token 不该通过 URL 传，通常应通过 Authorization header 传。**

## 第七句

**浏览器存 token 是高风险点，尤其是 refresh token。**

## 第八句

**Revoke 是“提前作废”，不是等它自然过期。**

## 第九句

**Refresh token rotation 是降低长期滥用风险的重要机制。**

## 第十句

**做 assessment 时，固定从 lifespan、format、storage/transport、revocation/rotation 四个维度检查 token。**

------

## 十四、一句话总结这章

你以后看到任何 token，都要从四个问题开始：
 **它活多久、长什么样、放在哪儿、出了问题怎么失效。**



# 第 10 部分：TLS、mTLS、证书、JKS、SSL bundle——传输层和证书层

这一章你先抓住一句话：

> **Token 保护“逻辑上的访问权限”，TLS/mTLS 保护“通信这条路本身”。**

也就是说：

- token 在回答：你能不能访问
- TLS 在回答：这条通信会不会被偷看、篡改、冒充

------

## 一、先说最基础的：为什么链路安全这么重要

假设一个前端或后端服务，已经拿到了合法的 access token。

如果通信链路不安全，会发生什么？

### 可能的问题

- token 在传输中被窃听
- 请求内容被篡改
- 响应内容被篡改
- 你以为连的是公司 API，实际上连到假服务
- 服务端以为请求来自合法客户端，实际上对方是伪装的

所以你要建立一个非常重要的直觉：

> **应用层安全做得再好，也不能替代传输层安全。**

这就是为什么 rules 里会单独有一类：

- TLS 必须 everywhere
- 证书链校验
- pinning
- mTLS
- client authentication based on certificates

------

# 二、TLS 是什么

------

## 1）最简单定义

**TLS** 是保护网络通信的加密协议。
 你可以把它理解成：

> **给通信加密、校验完整性、确认对方身份的标准方式。**

------

## 2）通俗理解

你可以把 TLS 想成两件事同时发生：

### 第一件：加密

别人即使截获了网络数据，也看不懂内容。

### 第二件：验明正身

客户端会检查：

- 我连接的这个服务，是不是真正的目标服务

------

## 3）和 HTTPS 的关系

这个你以后会经常看到。

### HTTP

明文通信。

### HTTPS

本质上就是：

> **HTTP + TLS**

所以你以后看到 HTTPS，就要想到：

- 这里底层用了 TLS
- 不是单纯“网页地址变成了 https”

------

## 4）为什么 TLS 在 API 项目里是基础要求

因为 API 传输的内容往往很敏感，比如：

- access token
- 用户数据
- 客户资料
- 账户信息
- 授权头
- session cookie

如果不用 TLS，这些东西在网络上暴露风险极高。

------

## 5）在 assessment 里什么时候会用到

你看到以下场景，就要想到 TLS：

- 浏览器到 gateway
- 前端到 BFF
- BFF 到 Apigee
- Apigee 到 backend
- backend 到 WebSSO / token endpoint
- 服务到服务调用

------

## 6）主要涉及哪些 rules

- **API0-R1**：最直接
- **API2-R7**：client authentication 里会延伸到 mTLS
- **API0-R2**：和标准授权链路搭配使用
- **API7-R4**：暴露面和安全通信也会间接相关

------

# 三、TLS 到底在保护什么

这个要讲清楚，不然后面你会把它只理解成“加密”。

------

## 1）保密性（Confidentiality）

别人看不懂传输内容。

### 例子

攻击者即使抓到了请求包，也看不到：

- token
- 用户数据
- API payload

------

## 2）完整性（Integrity）

别人不能悄悄改内容而不被发现。

### 例子

请求里原本是：

- 查询账户 A

攻击者不能在中间偷偷改成：

- 查询账户 B

而通信双方还毫无察觉。

------

## 3）服务端身份认证

客户端可以验证：

- 我连的是不是真的那个 gateway / API / WebSSO

### 例子

浏览器访问 `api.company.com` 时，浏览器会看证书是不是对应这个站点，是否由受信 CA 签发。

------

## 4）为什么这 3 个都重要

很多新人会只记“TLS=加密”，但在 assessment 里，你更要记住：

> **TLS 不只是防偷看，还防篡改，还防你连错人。**

------

# 四、证书（Certificate）是什么

------

## 1）最简单定义

**证书** 是一种数字身份证明，用来证明某个服务或客户端的身份。

### 通俗理解

你可以把它理解成：

> “网络世界里的身份证 / 盖章证明。”

------

## 2）它通常证明什么

常见场景里，它证明：

- 这个服务是 `api.company.com`
- 这个客户端是某个合法服务
- 这份公钥属于这个身份主体

------

## 3）为什么需要证书

因为客户端要验证：

- 我连的到底是不是目标服务

如果没有证书，攻击者更容易伪装成：

- 假 API
- 假 gateway
- 假 WebSSO

------

## 4）企业里常见的两种证书用途

### A. 服务端证书

最常见。
 比如：

- Apigee 暴露 HTTPS 地址
- WebSSO 暴露 token endpoint
- Spring Boot 服务暴露 HTTPS

客户端要验证服务端证书。

### B. 客户端证书

在 mTLS 场景里出现。
 比如：

- BFF 调内部高敏感 API
- gateway 调 backend
- 服务到服务双向认证

服务端也会要求客户端出示证书。

------

# 五、证书链（Certificate Chain）是什么

这个是企业里特别常见的检查点。

------

## 1）最简单理解

你不是只信“这一张证书”，而是要看：

> 这张证书是谁签的，签它的人我信不信，往上整条链条是不是可信。

这就叫证书链。

------

## 2）通俗比喻

像现实世界里：

- 这份证明是谁盖章的
- 给它盖章的机构是不是正规
- 这个机构是不是被更上层的权威认可

------

## 3）为什么 rules 会强调“server certification chain validation”

因为有些项目为了图省事，会关掉证书链校验，或者信任一切证书。
 这是很危险的。

如果证书链不严格校验，可能会导致：

- 连到假服务
- 被中间人攻击
- 测试环境的宽松配置跑进生产

------

## 4）在 assessment 里怎么问

你要看：

- 客户端是否默认做证书链校验
- 有没有 `trust all` 之类危险配置
- 是否有关闭 hostname verification
- RestTemplate / WebClient 是否用了不安全 SSL 配置

------

## 5）主要涉及哪些 rules

- **API0-R1**
- **API2-R7**

------

# 六、证书固定（Pinning）是什么

这个概念前面你也碰到过，现在正式讲一下。

------

## 1）最简单定义

**Certificate Pinning** 是指客户端不仅仅信任“任意受信 CA 签发的合格证书”，而是进一步绑定到特定证书或公钥。

### 通俗理解

你可以把它理解成：

> “我不光要看你有没有正规身份证，我还提前记住了你的这张身份证长什么样。”

------

## 2）为什么会需要 pinning

即使证书链是正规的，理论上仍可能发生某些风险场景：

- 错误签发
- 受信 CA 被滥用
- 中间人攻击窗口

pinning 能把信任范围再收窄。

------

## 3）为什么它常出现在“敏感 native app / browser-based app”规则里

因为这些场景面对的是更开放的网络环境。
 所以企业规则会说：

- 默认要做证书链校验
- 某些敏感场景还应考虑 pinning

------

## 4）在 assessment 里怎么问

你要看：

- 移动端是否实现 pinning
- 浏览器场景是否有企业规定
- 是否只是“理论上支持”，还是代码/SDK 真做了

------

## 5）主要涉及哪些 rules

- **API0-R1**

------

# 七、mTLS（Mutual TLS）是什么

这是这一章最核心的高级概念之一。

------

## 1）最简单定义

**mTLS = Mutual TLS = 双向 TLS**

### 普通 TLS

通常只有客户端验证服务端身份。

### mTLS

则是：

> **客户端验证服务端，服务端也验证客户端。**

------

## 2）通俗理解

普通 TLS 像这样：

- 你查验对方身份证
- 对方不一定查验你的身份证

mTLS 像这样：

- 双方都出示身份证
- 双方都检查对方身份

------

## 3）为什么企业里会喜欢 mTLS

因为它很适合高敏感的服务到服务场景：

- gateway → backend
- backend → backend
- internal API
- confidential client → token endpoint / API provider

### 它能提供什么

- 双向身份确认
- 更强的客户端认证
- 更适合 sender-constrained token 场景

------

## 4）什么时候特别适合 mTLS

一般是：

- internet-facing high sensitivity API
- internal sensitive service-to-service
- 高保密性 / 高完整性场景
- 想减少 bearer token 被偷后可直接复用的风险时

------

## 5）为什么 rules 会把 mTLS 和 JWT client authentication 放一起对比

因为它们都能做：

> **客户端认证**

但风格不同：

### JWT client authentication

更偏应用层，靠签名 token/assertion 证明 client 身份。

### mTLS

更偏网络层，靠客户端证书证明 client 身份。

------

## 6）在 assessment 里怎么用

看到 mTLS 时，你要问：

- 哪一段链路使用 mTLS
- 是 internet-facing 还是 internal
- 客户端证书是谁签发的
- 服务端是否真的验证客户端证书
- 证书 subject 是否和 client_id 有映射关系

------

## 7）主要涉及哪些 rules

- **API0-R1**
- **API2-R7**
- **API2-R1**

------

# 八、普通 TLS 和 mTLS 的区别，你要彻底记住

------

## 普通 TLS

### 典型场景

浏览器访问网站。

### 谁验证谁

- 浏览器验证服务端证书
- 服务端不一定验证浏览器证书

### 适合

- 大多数用户访问 API / 网站

------

## mTLS

### 典型场景

服务到服务、gateway 到 backend、高敏感 client auth。

### 谁验证谁

- 客户端验证服务端证书
- 服务端也验证客户端证书

### 适合

- internal APIs
- confidential client
- 高敏感场景

------

## 一句话记忆

- **TLS**：单向主要验服务端
- **mTLS**：双向互验

------

# 九、Keystore、Truststore、JKS：这些容器到底是干嘛的

这部分和你前面看到的代码、配置直接相关。

------

## 1）Keystore 是什么

### 最简单定义

保存“我自己的密钥材料”的地方。

### 里面常有什么

- private key
- 对应 certificate

### 通俗理解

你可以把它理解成：

> “我自己的证件和印章盒子。”

------

## 2）Truststore 是什么

### 最简单定义

保存“我信任谁”的证书集合。

### 里面常有什么

- trusted CA certificates
- trusted server certificates

### 通俗理解

你可以把它理解成：

> “我承认哪些机构/证件是可信的白名单。”

------

## 3）Keystore 和 Truststore 的区别

这个一定要记牢。

### Keystore

回答：

> 我拿什么证明我是我？

### Truststore

回答：

> 我凭什么相信对方是真的？

------

## 4）JKS 是什么

### 最简单定义

**JKS（Java KeyStore）** 是 Java 里常见的一种 keystore/truststore 文件格式。

### 通俗理解

你可以把它理解成：

> “Java 世界里装证书和私钥的保险箱文件。”

------

## 5）你前面代码里的 JKS 在干嘛

你前面那段 `ApimOauth` 代码里看到：

- keystore location
- key alias
- key password
- keystore password

这说明：

- 服务端在从 JKS 里取 key
- 然后用它去生成 assertion / token
- 再向 APIM / OAuth endpoint 证明 client 身份

这就是一个很典型的：

- confidential client
- 服务端密钥管理
- 应用级客户端认证

场景。

------

# 十、SSL bundle 是什么

这个是 Spring 层的概念，不是密码学新协议。

------

## 1）最简单定义

**SSL bundle** 是 Spring Boot 用来统一组织 SSL/TLS 配置的一种配置机制。

### 通俗理解

你可以把它理解成：

> “把证书文件、密码、类型、别名这些 SSL 相关配置打包成一个名字统一管理。”

------

## 2）它和 JKS 的关系

这个你要分清：

### JKS

是底层的证书/密钥容器文件。

### SSL bundle

是 Spring 配置层面对这些 SSL 资产的组织方式。

------

## 3）通俗比喻

- **JKS** 像“保险箱文件”
- **SSL bundle** 像“这套保险箱及其钥匙、密码、用途的配置说明卡”

------

## 4）为什么企业项目会喜欢 SSL bundle

因为 TLS 配置很容易散落在很多地方：

- RestTemplate
- WebClient
- 内嵌 Tomcat
- 外部调用
- mTLS 下游调用

用 bundle 可以更统一：

- 环境切换
- 配置治理
- 复用证书配置

------

## 5）在 assessment 里怎么用

你看到 `spring.ssl.bundle...` 时，不要紧张。
 你先知道：

- 项目在用 Spring 的 SSL 配置封装能力
- 底层可能是 JKS
- 这通常是配置治理上的正向信号

但还要继续问：

- 密钥文件放哪
- 密码怎么管理
- 是否只是测试配置
- 是否真的用于目标调用链路

------

# 十一、客户端证书（Client Certificate）是什么

------

## 1）它是什么

在 mTLS 场景里，客户端也要出示自己的证书。
 这个证书就叫客户端证书。

------

## 2）它解决什么问题

它帮助服务端确认：

> 来请求我的这个 client，真的是我认可的那个 client。

------

## 3）为什么企业规则会说“subject 要匹配 client_id”

因为如果只看“这是一张合法证书”还不够，
 系统通常还要知道：

- 这张证书到底属于哪个已注册 client

所以规则会要求：

- certificate subject ↔ client_id
   要有明确映射关系，而且通常在技术接入阶段预先登记好。

------

## 4）涉及哪些 rules

- **API2-R7**

------

# 十二、sender-constrained token 是什么（你先有个感觉）

这是稍高级一点的概念，但你现在需要先知道它为什么和 mTLS 连在一起。

------

## 1）先说 Bearer token 的问题

Bearer token 的特点是：

- 谁拿到谁就能用

所以如果 bearer token 被偷，攻击者常常能直接复用。

------

## 2）sender-constrained token 的直觉

它的思想是：

> 就算你偷到了 token，也不一定能用，因为这个 token 还绑定了发起方身份。

比如绑定到：

- mTLS 证书
- 某种密钥证明

------

## 3）为什么和 mTLS 常一起出现

因为 mTLS 可以天然提供“客户端是谁”的强绑定基础。
 所以企业规则会倾向说：

- 高敏感 internet-facing API
   更适合 mTLS + sender-constrained token
   而不是单纯 bearer token

------

## 4）你现在先怎么记

不用深究实现，先记一句：

> **mTLS 可以把 token 和具体客户端绑定得更紧，降低 bearer token 被偷后直接复用的风险。**

------

# 十三、这一章在 assessment 里怎么真正落地

以后你看到 TLS / 证书相关实现，不要只写一句“用了 HTTPS”。
 你要固定问下面这些问题。

------

## 1）这条链路有没有用 TLS

看：

- browser → gateway
- BFF → APIM
- service → service
- service → WebSSO/token endpoint

------

## 2）是否做了服务端证书校验

看：

- 是否有证书链校验
- 是否有 `trust all`
- 是否关闭 hostname verification

------

## 3）高敏感链路是否需要 mTLS

看：

- gateway → backend
- confidential client → API provider
- internal high-sensitivity APIs

------

## 4）客户端证书怎么管理

看：

- JKS / keystore
- alias
- password
- Vault / secret 管理
- 证书 subject 与 client_id 映射

------

## 5）配置是如何组织的

看：

- SSL bundle
- application.yml
- environment variables
- K8S secret

------

## 6）是否需要 pinning

尤其是：

- native app
- 某些 browser-based high sensitivity 场景

------

# 十四、这一章和 rules 的对应关系

------

## API0-R1 — API network security protocol

这一章最核心的 rule。

重点就是：

- TLS everywhere
- 证书链校验
- 敏感场景 pinning
- 符合企业密码要求

------

## API2-R7 — Client authentication

这一章第二核心 rule。

重点就是：

- JWT auth vs mTLS
- client certificate validation
- certificate subject ↔ client_id
- issuer / subject / audience / kid 校验

------

## API2-R1 — Grant flow

因为 confidential client 场景常常要和：

- mTLS
- JWT auth
- stronger client authentication

一起考虑。

------

# 十五、一个完整例子，把这一章串起来

假设一个 Spring BFF 要去向 APIM 申请 token，然后调用内部 API。

### 第一步：BFF 访问 token endpoint

这条链路必须用 TLS。

### 第二步：BFF 校验 APIM/WebSSO 的服务端证书

否则可能连到假服务。

### 第三步：如果是 mTLS，APIM 也会要求 BFF 出示客户端证书

这样 APIM 不只是“被访问”，还会确认“是谁在访问我”。

### 第四步：BFF 的私钥可能放在 JKS 里

然后通过 Spring SSL bundle 或其他配置方式组织起来。

### 第五步：后续 BFF 调内部高敏感 API 时，可能继续使用 mTLS

尤其在 gateway → backend 或 service → service 场景。

你看，这一整套都不是 token 本身，而是在保护“token 所走的路”。

------

# 十六、这一章最重要的 10 句话

## 第一句

**TLS 保护的是通信链路，不是访问权限本身。**

## 第二句

**TLS 不只是加密，还保护完整性和服务端身份认证。**

## 第三句

**HTTPS 本质上就是 HTTP over TLS。**

## 第四句

**证书是网络世界里的身份证明。**

## 第五句

**证书链校验的核心是：我信不信这张证背后的整条签发链。**

## 第六句

**mTLS 是双向认证：客户端验服务端，服务端也验客户端。**

## 第七句

**Keystore 放“我自己的密钥”，Truststore 放“我信任谁”。**

## 第八句

**JKS 是 Java 常见的证书/密钥保险箱文件格式。**

## 第九句

**SSL bundle 是 Spring 对 SSL 配置的组织方式，不是新的加密协议。**

## 第十句

**做 assessment 时，不要只写“用了 HTTPS”，而要继续问：链路是否全覆盖、证书是否严格校验、是否需要 mTLS。**

------

## 十七、一句话总结这章

你以后看到任何服务调用链路，都先问自己：
 **这条路是不是走 TLS、是不是验证了对方身份、在高敏感场景下是不是需要双向认证。**



# 第 11 部分：nonce、state、CSRF、replay attack

这一章你先抓住一句话：

> **安全不只是“内容合法”，还要保证“这是这一次该发生的请求”，而不是旧的、伪造的、被诱导的。**

------

## 一、先讲最根的问题：为什么“合法请求”也可能有问题

很多新人会直觉觉得：

- token 是真的
- 请求格式没错
- 用户也登录了

那不就安全吗？

不一定。

因为还有两类常见风险：

### 风险 1：重放攻击（Replay Attack）

攻击者把以前合法的一次请求、一次 token、一次认证结果偷过来，再用一次。

### 风险 2：伪造用户意图（CSRF）

攻击者诱导用户浏览器在“用户已登录”的状态下，帮攻击者发请求。

所以你会发现：

- 有些问题不是“身份不合法”
- 而是“时机不对、上下文不对、这不是用户这次真正想发的”

------

# 二、Replay Attack（重放攻击）是什么

------

## 1）最简单定义

**Replay Attack** 就是：

> 攻击者把之前一个合法的请求、凭证、认证结果复制下来，再次发送。

------

## 2）通俗理解

你可以把它理解成：

> 不是攻击者自己造假，而是把“你原来真的发过的那份东西”再拿出来用一次。

------

## 3）为什么它危险

因为被重放的内容往往本来就是真的：

- token 真的是授权服务器发的
- 请求格式真的是合法的
- 用户可能真的之前做过这个操作

所以系统如果只看“像不像真的”，很容易被骗。

------

## 4）几个银行/企业场景例子

### 例子 A：重放 access token

攻击者偷到了一个还没过期的 access token，又拿它去调 API。

### 例子 B：重放 authorization code

攻击者偷到了授权回调里的 code，尝试再去换 token。

### 例子 C：重放敏感操作请求

比如某次“确认提交”请求被抓到了，攻击者又重发一次。

### 例子 D：重放认证结果

攻击者用旧的 ID token / 认证响应冒充当前这次登录结果。

------

## 5）为什么 rules 会管 replay

因为 replay 不是单点问题，它会牵涉：

- token 生命周期
- code 是否一次性
- nonce
- nbf / exp
- 某些敏感请求是否有更强保护

------

# 三、nonce 是什么

------

## 1）最简单定义

**Nonce** 是一个一次性的随机值，用来保证：

> 这次认证结果对应的是这次请求，而不是旧结果被重放。

------

## 2）通俗理解

你可以把 nonce 理解成：

> “这次请求专属的一次性暗号”

客户端发起认证时先生成这个暗号，
 认证系统返回结果时，必须把这个暗号带回来。
 客户端再检查：

- 带回来的 nonce
- 是不是和自己刚才发出去的一样

如果不一样，就说明有问题。

------

## 3）为什么 nonce 能防 replay

因为攻击者即使偷到了旧的认证结果，也会遇到问题：

- 旧结果里的 nonce 是旧的
- 客户端现在等的是这次新生成的 nonce

所以旧结果不能冒充当前这次。

------

## 4）nonce 最常出现在哪儿

在你当前任务里，最典型是：

- **OIDC 认证请求**
- **ID token 校验**

也就是说，nonce 特别常用于：

> **“防止旧的认证结果冒充新的认证结果”**

------

## 5）你在项目里会在哪里看到 nonce

### 前端 / OIDC 客户端代码

- login request generation
- auth request parameters
- callback validation

### OIDC library 配置

- nonce storage
- nonce verification

### 安全日志 / token 解析逻辑

- ID token validation step

------

## 6）和哪些 rules 相关

- **API6-R1**：最直接
- **API2-R6**：Authorization Code / OIDC 安全细节
- 也和 **API0-R2** 有间接关系

------

# 四、state 是什么

很多人把 nonce 和 state 混在一起，这里一定分开。

------

## 1）最简单定义

**State** 是客户端在发起认证请求时带上的一个值，用来保证：

> 回来的这个响应，真的是我刚刚发起的那一次请求的响应。

------

## 2）通俗理解

你可以把它理解成：

> “这是我自己刚才开的那张工单号，回来时必须对得上。”

------

## 3）为什么需要 state

因为在浏览器跳转型流程里，客户端会把用户带去登录，然后再被重定向回来。
 如果没有一个绑定值，客户端很难确认：

- 回来的这次回调
- 到底是不是自己之前发起的那次

这就给了攻击者空间，去伪造或混淆回调流程。

------

## 4）state 最主要防什么

它最主要是为了防：

- **CSRF**
- request/response mismatching
- 登录流程被劫持或串改

------

## 5）你在项目里会在哪里看到 state

### 前端 / 浏览器登录流程

- 发起登录前生成 state
- 回调回来后验证 state

### BFF / Web 应用

- session 里存 state
- callback handler 里比对 state

------

## 6）和哪些 rules 相关

- **API7-R5**：最直接
- **API2-R6**：也会涉及认证流程安全细节

------

# 五、nonce 和 state 到底有什么区别

这是这一章最关键的区别之一。

------

## 1）先给一句最简单的区分

### state

更偏：

> **防伪造回调 / 防 CSRF / 确保这是我发起的那次请求**

### nonce

更偏：

> **防认证结果被重放 / 确保返回的是这次新鲜结果**

------

## 2）更通俗地记

### state

你可以记成：

- “这是不是我开的工单”

### nonce

你可以记成：

- “这是不是这次刚生成的一次性暗号”

------

## 3）为什么两者不能混用

因为它们保护的侧重点不同：

- state 更强调请求和回调是否配对
- nonce 更强调认证结果是不是旧的被重放

所以企业里经常两个都要。

------

# 六、CSRF 是什么

------

## 1）最简单定义

**CSRF（Cross-Site Request Forgery）** 是一种攻击：
 攻击者诱导用户浏览器，在用户已登录的情况下，向目标系统发送用户本来没想发的请求。

------

## 2）通俗理解

你可以把它理解成：

> “用户登录着银行系统，但攻击者骗浏览器偷偷替用户按了按钮。”

用户自己未必意识到请求已经发出去了。

------

## 3）为什么浏览器场景特别容易有 CSRF

因为浏览器会自动带上一些状态，比如：

- cookie
- 某些会话信息

如果系统只靠 cookie 判断“用户已登录”，那浏览器可能在用户不知情时自动带着 cookie 发请求。

------

## 4）一个直觉例子

### 场景

用户已经登录一个系统，浏览器里有有效 session cookie。

### 风险

攻击者诱导用户打开某个恶意页面，这个页面偷偷触发请求到目标系统。

### 如果系统只看 cookie

它会以为：

- 这是合法用户本人发的请求

但实际上：

- 这是浏览器被诱导发的请求

------

## 5）为什么这和 cookie/session 模式特别相关

因为 cookie 会自动跟请求一起发出去。
 所以只要系统把“有 cookie = 用户本人主动操作”画等号，就容易出事。

这也是为什么你前面的 rule API7-R5 专门提：

- `state` 参数
- 而你前面文档里还备注了“cookie-managed API 还要额外考虑 CSRF”

------

## 6）在 assessment 里怎么理解

你要先看这个项目是不是：

- browser-based
- cookie/session 模式
- 有敏感状态改变操作

如果是，就必须认真想 CSRF，不是只盯 token。

------

## 7）和哪些 rules 相关

- **API7-R5**
- **API2-R6**
- **API2-R8**
- **API0-R2**（BFF/cookie 模式时也相关）

------

# 七、One-time use（一

次性使用）为什么重要

这也是防 replay 的关键思想。

------

## 1）最简单定义

**One-time use** 就是：

> 这个凭证/这个值只能用一次，用完就作废。

------

## 2）典型例子

### Authorization Code

用一次换完 token 就应该失效。

### Refresh Token Rotation

每次 refresh 后旧的 refresh token 失效。

### State / Nonce

这次请求用完，下次不能再复用。

------

## 3）为什么它对防 replay 特别有效

因为 replay 的本质就是“拿旧的东西再用一次”。
 如果旧东西一旦被用过就作废，那重放空间就小很多。

------

## 4）和哪些 rules 相关

- **API2-R2**
- **API2-R5**
- **API2-R6**
- **API6-R1**
- **API7-R5**

------

# 八、Freshness（新鲜度）是什么

这个词 rules 里未必总直接写出来，但你必须有这个概念。

------

## 1）最简单理解

**Freshness** 就是：

> 我怎么知道这次结果是“刚刚这次产生的”，而不是历史遗留结果。

------

## 2）哪些机制在帮助保证 freshness

- nonce
- state
- `exp`
- `nbf`
- one-time use
- 短时 token/code
- refresh token rotation

------

## 3）为什么它重要

因为系统不光要验证“这个东西真假”，还要验证：

- 它是不是还属于当前这次上下文
- 它是不是已经过时了
- 它是不是被用过了

------

# 九、Replay Attack 和 Token 生命周期、Code、Rotation 的关系

这一块是把你前两章串起来。

------

## 1）短生命周期为什么能减轻 replay

如果 access token 只活很短时间，即使被偷：

- 可利用时间窗口也小

这就是为什么 **API2-R2** 和 **API6-R1** 是连在一起理解的。

------

## 2）Authorization Code 一次性为什么重要

因为如果 code 不一次性，攻击者偷到了旧 code 还可能继续换 token。

------

## 3）Refresh Token Rotation 为什么重要

因为如果 refresh token 固定长期不变，被偷以后攻击者能持续续命。
 rotation 可以更快暴露重复使用的异常。

------

## 4）state / nonce 为什么和 flow 安全绑定

因为它们让：

- 当前请求
- 当前回调
- 当前认证结果

彼此更紧地绑定，减少旧结果和伪造结果被混进去的机会。

------

# 十、你在项目里怎么真正检查这些东西

这部分特别实用。

------

## 1）检查 nonce

你要看：

### 前端 / 客户端

- 发起 OIDC 登录时是否带 nonce
- callback 后是否验证 nonce
- nonce 是否一次性、是否随机

### BFF

- 是否在服务端 session 中保存 nonce
- 返回后是否比对

------

## 2）检查 state

你要看：

- 是否发起登录时生成 state
- 是否在 callback 时验证 state
- state 是否是随机、一次性、session-specific

------

## 3）检查 replay 防护

你要看：

- access token 是否短命
- code 是否 one-time use
- refresh token 是否 rotation
- JWT auth token/assertion 是否很短
- 敏感请求是否有额外保护

------

## 4）检查 CSRF 风险

你要先判断项目是否：

- 浏览器场景
- cookie/session 管理
- 有状态修改操作

然后看：

- 是否有 state / anti-CSRF 机制
- 是否只是“有 cookie 就认定合法”

------

# 十一、你在代码里会在哪里看到这些东西

------

## 1）前端 / OIDC 客户端库

找：

- `state`
- `nonce`
- callback validation
- auth request builder

------

## 2）BFF / Spring Security / session 逻辑

找：

- state/nonce 存储位置
- redirect callback handler
- logout / session invalidation
- anti-CSRF 配置

------

## 3）WebSSO / IdP 配置

找：

- 是否要求 nonce
- flow 配置
- redirect validation

------

## 4）Token / refresh 流程

找：

- refresh token rotation
- revoke
- one-time use
- expiration

------

# 十二、这一章和 rules 的对应关系

------

## API6-R1 — Replay attacks

这是本章最核心的 rule。

重点关注：

- nonce
- 短时 token
- 敏感请求的额外保护

------

## API7-R5 — CSRF attack

重点关注：

- state
- one-time use
- session-specific
- unpredictable

------

## API2-R6 — OIDC Authorization Code Grant specifics

重点关注：

- redirect_uri
- PKCE
- 流程安全细节
- 和 state/nonce 形成配套

------

## API2-R2 / API2-R5

它们虽然不是直接讲 replay/CSRF，但会从：

- lifespan
- one-time use
- rotation
   这些角度支撑 replay 防护。

------

# 十三、一个完整例子，把这一章串起来

假设一个 Angular 前端通过 OIDC 登录。

### 第一步

前端生成一个随机 `state` 和一个随机 `nonce`。

### 第二步

前端把用户带到 WebSSO 登录，并带上这两个值。

### 第三步

用户登录成功，浏览器被重定向回来，回调里带着 code 和 state。

### 第四步

客户端先检查：

- 回来的 `state`
- 是否和自己发起时保存的一样

这一步主要在防伪造回调 / CSRF。

### 第五步

客户端再用 code 换到 ID token / access token。

### 第六步

客户端检查 ID token 里的 `nonce`

- 是否和之前发出的 nonce 一致

这一步主要在防旧认证结果被重放。

### 第七步

access token 本身又是短时有效的，即使泄露，窗口也比较小。

你看，这一整套组合拳才是完整安全，不是某一个参数单独包打天下。

------

# 十四、这一章最重要的 9 句话

## 第一句

**Replay attack 不是伪造新内容，而是重用旧的合法内容。**

## 第二句

**Nonce 是一次性随机值，主要用来防认证结果被重放。**

## 第三句

**State 主要用来绑定认证请求与回调，防伪造回调和 CSRF。**

## 第四句

**State 和 nonce 不是一回事，通常两者都需要。**

## 第五句

**CSRF 本质上是诱导浏览器带着用户现有登录状态发出用户并不想发的请求。**

## 第六句

**Cookie/session 模式下尤其要认真考虑 CSRF。**

## 第七句

**One-time use、短时有效、rotation 都是在帮助防 replay。**

## 第八句

**Freshness 的核心是：证明这次结果属于这次请求，而不是旧结果。**

## 第九句

**做 assessment 时，不要只看“登录能不能成功”，还要看请求和回调是不是被正确绑定、是不是能防重放。**

------

## 十五、一句话总结这章

你以后看到登录流程或敏感请求时，都先问自己：
 **这是不是这一次该发生的请求/结果，而不是旧的、伪造的、被诱导的。**



# 第 12 部分：API Gateway / Apigee / Axway 到底管什么，不管什么

这一章你先抓住一句话：

> **Gateway 是“前门”，不是“整栋楼”。**

也就是说：

- 它能决定你能不能先进楼
- 但它不一定知道你进楼后能不能打开 307 房间的保险柜

这个区别，是你做 assessment 时最重要的一个思维分界。

------

## 一、先说 API Gateway 到底是什么

### 1）最简单定义

**API Gateway** 是系统对外暴露 API 时的统一入口层。

### 2）通俗理解

你可以把它理解成：

> 外部请求进系统之前，先经过的总闸门。

它像一个统一接待台，负责：

- 看请求从哪来
- 看请求是否符合规则
- 看请求该去哪里
- 先挡掉一批明显不合规、不该进来的请求

------

## 二、Apigee、Axway 和 API Gateway 是什么关系

### 1）API Gateway 是角色 / 类别

它是“这一层系统”的通用叫法。

### 2）Apigee、Axway 是具体产品

你可以这样理解：

- **API Gateway** 是“职位名称”
- **Apigee / Axway** 是“具体员工名字”

### 3）在你当前任务里怎么先记

- **Apigee**：你们现在主要的 API 管理 / gateway 平台
- **Axway API Manager**：你们之前用过的 gateway / API management 平台
- 它们都属于“API Provider / API Gateway”这一层

------

## 三、为什么企业一定喜欢有 Gateway

因为如果没有统一 gateway，常见会出这些问题：

### 1）每个后端自己裸奔

- 各自暴露地址
- 各自做一套鉴权
- 各自做一套限流
- 各自做一套错误处理

结果就是：

- 不统一
- 不好审计
- 不好治理
- 很容易漏

------

### 2）外部流量直接碰后端

这会带来：

- 暴露面大
- 安全基线不统一
- 难以做集中控制
- 不符合很多企业架构要求

------

### 3）很难统一做这些事情

- token 基础校验
- rate limiting
- CORS
- header 清洗
- routing
- 监控
- API inventory
- 版本与产品化管理

所以企业喜欢 gateway，不是因为“时髦”，而是因为：

> **它特别适合把一批通用的 API 安全与治理动作，集中放在前门做。**

------

# 四、Gateway 最擅长做什么

这一部分你后面做 assessment 会反复用到。

------

## 1）统一入口（Exposition Layer）

### 通俗理解

所有 external-facing API 先经过 gateway。

### 为什么重要

这样才能确保：

- 外部流量先过同一道门
- 不会有人绕过门卫直接摸进后端

### 对应 rule

- **API0-R3**

这是 gateway 最强、最天然的一条。

------

## 2）路由（Routing）

### 通俗理解

请求进来后，gateway 决定把它转发给哪个后端服务。

### 例子

- `/customer/*` → customer-service
- `/loan/*` → loan-service

### 为什么重要

因为路由不只是“转发”，还决定：

- 哪些路径对外暴露
- 后端真实地址是否被隐藏
- 是否存在不该暴露的技术接口

### 对应 rule

- **API0-R3**
- **API7-R4**
- **API3-R3**

------

## 3）粗粒度授权（Coarse-grained Authorization）

### 通俗理解

先判断你大方向能不能访问这类 API。

### 例子

- 你的 token 有没有 `read_accounts`
- 这个 client_id 有没有权调用这个 API product

### 为什么适合 gateway 做

因为它擅长处理：

- token
- scope
- API product
- client app
- policy

### 对应 rule

- **API1-R1**（粗粒度部分）
- **API5-R1**
- **API3-R4**

------

## 4）Token 基础校验

### 通俗理解

先看这张票像不像真的。

### 常见检查

- token 是否存在
- 是否过期
- issuer 对不对
- audience 对不对
- signature 对不对
- scope 是否够

### 为什么适合 gateway 做

因为这是“门口验票”的事。

### 对应 rule

- **API0-R2**
- **API2-R3**
- **API2-R7**
- **API5-R1**

------

## 5）Rate Limiting / Quotas

### 通俗理解

限制某个 client / user / IP 在单位时间内能发多少请求。

### 为什么特别适合 gateway 做

因为它是所有流量必经之地，很适合统一计数和统一拦截。

### 对应 rule

- **API4-R1**

这也是 gateway 最典型、最强的一项能力。

------

## 6）CORS 控制

### 通俗理解

如果 API 会被浏览器跨域调用，gateway 可以统一控制：

- 哪些 origin 允许
- 哪些 method 允许
- 哪些 header 允许

### 为什么适合 gateway 做

因为它就在浏览器请求进入系统的最前面。

### 对应 rule

- **API7-R1**
- **API3-R4**（avoid `Origin *` 也有关）

------

## 7）Header 处理

### 通俗理解

gateway 可以统一：

- 添加某些头
- 删除某些头
- 清洗内部头
- 加安全头

### 为什么重要

因为很多信息泄露问题和兼容问题都在 header 层。

### 对应 rule

- **API7-R3**
- **API7-R2**
- **API10-R1**（部分日志与追踪头也有关）

------

## 8）请求基础校验（Syntactic Validation）

### 通俗理解

先看请求长得像不像一个合法请求。

### 例子

- method 对不对
- path 对不对
- content-type 对不对
- body schema 对不对
- 参数格式对不对

### 为什么适合 gateway 做

因为这类校验是通用的、统一的，不一定需要懂业务细节。

### 对应 rule

- **API8-R1**

------

## 9）错误响应基线处理

### 通俗理解

统一避免返回太多技术细节。

### 例子

- 不把堆栈直接吐给外部
- 不把内部服务名、版本、路径暴露出去
- 用统一格式返回错误

### 对应 rule

- **API7-R2**

------

## 10）日志、监控、追踪（部分）

### 通俗理解

gateway 可以统一记录：

- 谁在调 API
- 调了什么路径
- 是不是被限流了
- token 是否无效
- 来源 IP 是谁

### 为什么重要

因为它是全局视角的观察点。

### 对应 rule

- **API10-R1**
- **API7-R6**
- **API7-R7**（测试时也常看这里）

------

# 五、Gateway 不擅长做什么，或者不应该单独负责什么

这一部分比“它能做什么”更重要。
 因为很多 assessment 的错误，就出在把 gateway 想得太万能。

------

## 1）它不应该替代细粒度业务授权

### 通俗理解

gateway 可能知道你有 `read_accounts` scope，
 但它通常不知道：

- 账户 123 到底是不是张三的
- 这个客户是不是归当前员工负责
- 这个字段是不是只允许某类角色看

### 为什么

因为这些判断依赖：

- 业务数据
- DB 查询
- 领域规则
- 用户与对象关系

这些通常只有业务后端最清楚。

### 对应 rule

- **API1-R1**
- **API3-R1**
- **API3-R4**

------

## 2）它不应该单独决定对象级授权（Object-level Authorization）

### 例子

请求：

```
GET /accounts/123
```

gateway 可能知道你能访问 `/accounts/*`
 但它不知道：

- `123` 是不是你的账户

所以如果项目只靠 gateway 放行，而后端不再检查对象 ownership，就会出问题。

### 这就是为什么 BOLA / IDOR 常发生在 backend，不是 gateway 单独能解决的。

### 对应 rule

- **API1-R1**
- **API3-R1**
- **API7-R4**

------

## 3）它不应该单独决定字段级授权

### 例子

后端返回 customer profile 时：

- 哪些字段给 customer 看
- 哪些字段给 advisor 看
- 哪些字段不应返回给前端

gateway 最多能做一些 response filtering，但它通常不最懂业务语义。

### 对应 rule

- **API3-R1**
- **API3-R2**
- **API1-R1**

------

## 4）它不负责真正的业务输入语义校验

### 区分两个层次

#### Gateway 擅长

- 参数类型对不对
- 格式对不对
- schema 对不对

#### Backend 必须做

- 这个合同号在业务上合不合理
- 这个账户是不是属于当前用户
- 这个状态变更是否允许

也就是：

- gateway 做 **syntactic validation**
- backend 做 **semantic/business validation**

### 对应 rule

- **API8-R1**

------

## 5）它不负责完整的 session / SSO 生命周期管理

### 通俗理解

gateway 可以校验 token，但它通常不是：

- 登录系统本身
- WebSSO 会话管理者
- 用户 session 生命周期的真正所有者

### 所以这些东西你主要不该去 gateway 找答案

- SSO idle timeout
- MFA policy
- logout 后 session 是否真正失效
- password change 是否踢掉旧 session

### 对应 rule

- **API2-R4**
- **API2-R8**
- **API2-R9**

------

## 6）它不替代治理流程

gateway 可能帮助你管理 API，但很多治理要求不是“平台有功能”就自动满足。

### 例如

- API owner 是否明确
- unused API 是否 6 个月停用
- API registry 是否完整
- major release 前是否做 pentest
- SAST 报告是否通过

这些很多是流程、治理、组织责任问题，不是 gateway 单独搞定。

### 对应 rule

- **API7-R6**
- **API7-R7**
- **API8-R2**
- **API9-R1**
- **API9-R2**

------

# 六、你可以把 Gateway 想成“门口保安”，但不要想成“整个公司”

这个比喻很适合你当前阶段。

------

## Gateway 像门口保安能做什么

- 看你有没有门票
- 看你门票有没有过期
- 看你该去哪个楼层
- 看你是不是来得太频繁
- 不让你带明显违规的东西进来

------

## 但保安做不了什么

- 他不知道你是不是这个保险柜的真正主人
- 他不知道合同 123 背后的业务关系
- 他不知道这个字段是不是只能给风控人员看
- 他不替你决定整个公司的人事制度和审计制度

------

# 七、为什么你前面那个 Apigee guideline 只能覆盖部分 rules

现在你应该能更好理解了。

因为那个 guideline 很强地覆盖了 gateway 擅长的部分：

- proxy configuration
- rate limit
- CORS
- error handling
- input validation
- token validation
- monitoring

但对下面这些，它天然就不可能完整覆盖：

- session lifecycle
- SSO lifespan
- MFA policy
- object-level authorization
- field-level need-to-know
- client_id 唯一性治理
- unused API retirement

所以你前面那个结论“它只能部分覆盖整套 enterprise rules”是很合理的。

------

# 八、在项目里，去哪儿看 Gateway 做了什么

这部分很实用。

------

## 1）看 Apigee / Axway policy

你重点找这些：

- VerifyJWT / OAuth verify
- quota / spike arrest / rate limit
- CORS policy
- header manipulation
- assign message / fault rule
- routing / target endpoint
- logging / analytics

------

## 2）看 API product / app 配置

看：

- 哪些 client app 被允许
- 哪些 scope/API 被绑定
- 是否有限流策略
- 是否按产品暴露

------

## 3）看 OpenAPI / Swagger

看：

- 对外暴露的是不是 gateway 域名
- 是否通过 gateway path 暴露
- 有没有不该对外暴露的 technical endpoints

------

## 4）看架构图和部署入口

看：

- 外部流量是不是先到 gateway
- backend 有没有直连入口
- ingress / route / LB 是否只指向 gateway

------

# 九、做 assessment 时，关于 Gateway 你要固定问的 8 个问题

以后看到项目用了 Apigee / gateway，就固定问：

### 1. 外部流量是否全部经过 gateway？

对应：

- API0-R3

### 2. Gateway 是否在做 token 基础校验？

对应：

- API0-R2
- API2-R7
- API5-R1

### 3. Gateway 是否配置了 rate limiting / quota？

对应：

- API4-R1

### 4. Gateway 是否正确限制 CORS？

对应：

- API7-R1
- API3-R4

### 5. Gateway 是否统一处理安全 headers / error responses？

对应：

- API7-R2
- API7-R3

### 6. Gateway 是否做了请求基础校验？

对应：

- API8-R1

### 7. Gateway 是否暴露了不该暴露的路径或技术接口？

对应：

- API3-R3
- API7-R4

### 8. 项目是否错误地把所有授权责任都推给 gateway？

对应：

- API1-R1
- API3-R1
- API3-R4

------

# 十、把 Gateway 和 Backend 的分工再定一次

这个一定要牢。

------

## Gateway 更像负责

- 统一入口
- 基础验票
- 粗粒度控制
- 流量治理
- 协议与安全基线
- 暴露面治理

------

## Backend 更像负责

- 业务规则
- 对象级授权
- 字段级授权
- need-to-know
- secret data exposure
- 语义校验
- 最终数据返回边界

------

# 十一、这一章和 rules 的对应关系

------

## Gateway 强相关、适合重点在 gateway 找证据的

- **API0-R3**
- **API4-R1**
- **API7-R1**
- **API7-R2**
- **API7-R3**
- **API7-R4**
- **API8-R1**
- **API10-R1**（部分）

------

## Gateway 部分相关，但不能只看 gateway 的

- **API0-R2**
- **API1-R1**
- **API3-R1**
- **API3-R4**
- **API5-R1**

------

## 不应主要靠 gateway 解决的

- **API2-R4**
- **API2-R8**
- **API2-R9**
- **API9-R1**
- **API9-R2**

------

# 十二、一个完整例子，把这一章串起来

假设外部用户调用客户查询 API。

### 第一步：请求先到 Apigee

Apigee 检查：

- token 是否有效
- scope 是否允许
- rate limit 是否超了
- CORS 是否允许
- path 是否是合法暴露路径

### 第二步：Apigee 转发给 customer-service

这时 gateway 已经做了第一轮通用控制。

### 第三步：customer-service 再判断

- 这个 customerId 是不是当前用户可访问的
- 是否只能返回部分字段
- 是否隐藏敏感数据

你看：

- Gateway 负责“先挡住不该进的”
- Backend 负责“进来之后到底能看到什么”

这就是完整的企业分工。

------

# 十三、这一章最重要的 10 句话

## 第一句

**Gateway 是前门，不是整个系统。**

## 第二句

**Apigee/Axway 是具体产品，API Gateway 是这一层角色。**

## 第三句

**Gateway 最擅长统一入口、粗粒度控制、流量治理和安全基线。**

## 第四句

**Gateway 特别适合做 token 基础校验、rate limiting、CORS、header/error 基线处理。**

## 第五句

**Gateway 可以做请求的基础格式校验，但不替代业务语义校验。**

## 第六句

**Gateway 不应单独承担对象级授权和字段级授权。**

## 第七句

**Need-to-know 和 fine-grained authorization 最终要落到 backend。**

## 第八句

**Session、SSO、MFA 这类问题通常不该主要去 gateway 找答案。**

## 第九句

**有 gateway 不等于所有 API security rules 都自动满足。**

## 第十句

**做 assessment 时，要先判断这条 rule 是不是 gateway 该负责的那一层。**

------

## 十四、一句话总结这章

你以后看到 Apigee / API Gateway 时，先问自己：
 **它是在做“前门该做的事”，还是项目错误地把“不属于前门的事”也都指望它来解决。**



# 第 13 部分：输入校验、输出控制、CORS、rate limit、error handling

这一章你先抓住一句话：

> **API 安全不只是“谁能进”，还包括“进来的人能发什么、系统会回什么、回得有多克制”。**

也就是说，前面几章更多在讲：

- 谁是谁
- token 是什么
- gateway 干什么

这一章开始落到 API 最日常、最具体的行为上。

------

## 一、先给你一张总图：这一章其实在管 5 件事

你后面做 assessment 时，这一章基本可以拆成 5 个固定问题：

### 1. 输入校验（Input Validation）

请求进来时，系统有没有检查：

- 长得对不对
- 类型对不对
- 值合不合理

### 2. 输出控制（Output Control）

响应出去时，系统有没有避免：

- 给太多字段
- 暴露敏感信息
- 把内部细节带出去

### 3. CORS

如果浏览器跨域调 API，系统有没有只放行该放行的前端来源。

### 4. Rate Limiting

有没有限制谁可以在单位时间里调用多少次。

### 5. Error Handling

出错时，系统有没有控制错误信息，不把内部实现细节暴露给外部。

你可以把这一章理解成：

> **API 的门口安检、前台接待、对外说话方式，全都在这里。**

------

# 二、输入校验（Input Validation）是什么

------

## 1）最简单定义

**输入校验** 就是：

> 系统收到请求后，先检查这个请求是不是一个“合法、合理、可接受”的请求。

------

## 2）通俗理解

你可以把它理解成：

> “不是所有看起来像请求的东西，都应该交给业务逻辑处理。”

系统先要做一道过滤：

- 请求格式对不对
- 参数类型对不对
- 值域合不合理
- 有没有明显恶意内容

------

## 3）为什么输入校验很重要

因为很多攻击本质上不是“身份错了”，而是：

- 传了奇怪参数
- 传了非法格式
- 传了恶意内容
- 传了超长、超大、超出预期的数据

如果系统不拦，后果可能是：

- SQL injection
- command injection
- XSS 载荷进入后端
- 业务逻辑异常
- 系统崩溃
- 敏感信息泄露

------

# 三、输入校验其实分两层：语法校验 和 语义校验

这个一定要分开，因为它直接对应 gateway 和 backend 的分工。

------

## 1）Syntactic Validation（语法层校验）

### 通俗定义

先判断这个请求“长得像不像一个合法请求”。

### 例子

- method 是否允许
- path 是否正确
- `Content-Type` 是否正确
- body 是否符合 JSON schema
- 参数是不是数字、日期、布尔
- 字段长度是否超限
- 枚举值是否在白名单里

### 谁更擅长做

通常：

- API Gateway
- OpenAPI / schema 校验层
- Spring 的基础参数校验

------

## 2）Semantic Validation（语义/业务校验）

### 通俗定义

即使请求“长得合法”，还要判断它“业务上合不合理”。

### 例子

- 这个 accountId 是否属于当前用户
- 这个合同号格式虽然对，但是否真的存在
- 这个状态能不能从 A 改到 B
- 这个金额是否超过业务允许范围
- 这个客户是不是当前员工可操作的对象

### 谁更擅长做

通常只能是：

- Business Service
- 业务后端
- service + domain logic + DB 约束

------

## 3）为什么这两层都要有

如果只有语法校验：

- 请求看起来合法
- 但业务上可以是恶意的

如果只有语义校验：

- 很多明显垃圾请求早该在门口拦住，却一路走进后端

所以企业里通常要求：

> **Gateway 做 syntactic validation，Backend 再做 semantic validation。**

------

# 四、输入校验具体会查什么

你后面看项目时，可以按下面这张清单去看。

------

## 1）HTTP Method

### 问什么

这个接口只允许：

- GET
- POST
- PUT
- DELETE

中的哪几个？

### 风险

如果本来只该 GET，却其他 method 也放进来了，可能扩大攻击面。

------

## 2）Path / Endpoint

### 问什么

这个路径是不是 API 设计允许的路径？

### 风险

- 非预期路径被暴露
- 经典路径容易被猜
- 技术接口暴露出去

### 对应 rule

- **API7-R4**
- **API3-R3**

------

## 3）Content-Type

### 问什么

系统是不是只接受预期的 content type？

### 例子

只接受：

- `application/json`

不应随便接受：

- 不该有的 multipart
- 奇怪文本格式
- 混乱编码

### 对应 rule

- **API7-R3**
- **API8-R1**

------

## 4）参数类型

### 问什么

这个字段到底应该是：

- string
- integer
- boolean
- date
- array
- object

### 风险

如果不严格，攻击者可能传非常规类型或特殊结构，绕过逻辑。

------

## 5）长度与范围

### 问什么

- 字符串最长多长
- 数值最小/最大多少
- 数组最多多少项
- 文件最大多大

### 风险

- 超长 payload
- 资源消耗
- 溢出/异常
- 业务绕过

------

## 6）Pattern / 格式白名单

### 问什么

这个字段是不是应该只允许某种格式？

### 例子

- email 格式
- date 格式
- contract number 格式
- 只允许特定字符集

### 风险

如果任意字符都放进来，注入和格式绕过风险会增大。

------

## 7）自由文本（Free-form text）

### 问什么

如果这是用户自由输入文本，系统有没有做必要清洗？

### 风险

- HTML 注入
- JavaScript 载荷
- SQL 片段
- 特殊控制字符

### 对应 rule

- **API8-R1**

------

## 8）文件上传

### 问什么

- 是否限制文件类型
- 是否限制大小
- 是否扫描恶意内容
- 是否控制存储位置

### 风险

文件上传是企业里很常见的高风险输入面。

### 对应 rule

- **API8-R1**

------

# 五、Schema Validation 是什么

------

## 1）最简单定义

**Schema Validation** 就是：

> 用一份“请求/响应结构说明书”去检查实际数据是不是符合约定。

------

## 2）通俗理解

你可以把它理解成：

> “系统先拿着模板对照，看这份 JSON 有没有按模板来。”

比如某个接口要求：

- `customerId` 必填
- `amount` 必须是 number
- `currency` 必须是三位字符串
- `comments` 最长 200 字

那 schema validation 就是拿这些规则去自动校验请求。

------

## 3）为什么它特别适合 gateway

因为这是典型的：

- 通用
- 结构性
- 可自动化

的检查，不一定依赖业务数据。

------

## 4）你在项目里会在哪里看到它

- OpenAPI / Swagger
- Apigee policy
- JSON schema
- Spring validation annotations（部分）
- request DTO 校验

------

## 5）对应哪些 rules

- **API8-R1**
- 也会帮助 **API3-R1**（因为输出结构也能控制）

------

# 六、输出控制（Output Control）是什么

这部分和输入校验一样重要，而且很多新人会忽略。

------

## 1）最简单定义

**输出控制** 就是：

> 系统返回响应时，只返回该返回的内容，不多给，不乱给。

------

## 2）为什么它重要

因为很多泄露不是输入打穿系统，而是：

- 后端本来就把太多东西返回出来了
- 前端只是“没显示”，但数据其实已经到了浏览器
- 攻击者一抓包就全看到了

------

## 3）最常见的两个问题

### A. Excessive Data Exposure（过度数据暴露）

后端返回了远超前端实际需要的字段。

### B. Secret / Sensitive Data Exposure

后端把本不该对外暴露的敏感字段直接返回了。

------

# 七、Need-to-know 在输出控制里怎么落地

前面你已经学过 **need-to-know**，这里把它落地到“响应”层。

------

## 1）最简单理解

不是用户登录了，就把整个对象全返回。
 而是：

> **这个调用方为了完成当前职责，到底需要看到哪些字段。**

------

## 2）银行场景例子

客户账户页面只需要：

- maskedAccountNumber
- accountType
- balance
- currency

那后端就不应该顺手再返回：

- internalRiskScore
- fullCustomerProfile
- operatorNotes
- internalFlags

------

## 3）为什么前端“没展示”不算安全

因为：

- 浏览器仍然能抓包
- 调试工具仍然能看到
- 恶意脚本仍然可能读到

所以真正的控制点必须在：

- backend response
- 或 gateway response filtering

而不是“UI 没显示”。

------

## 4）对应哪些 rules

- **API3-R1**
- **API3-R2**
- **API1-R1**
- **API3-R4**

------

# 八、Mass Assignment 是什么

这个在前面 Apigee 文档里也出现过，这里补进去。

------

## 1）最简单定义

**Mass Assignment** 是指：

> 客户端多传了一些本不该允许它控制的字段，而后端不小心全接收了。

------

## 2）通俗理解

你可以把它理解成：

> “客户端偷偷多塞了几个字段，后端居然照单全收。”

------

## 3）典型例子

前端本来应该只能传：

- name
- email

结果攻击者自己构造 JSON，多加：

- `isAdmin: true`
- `status: APPROVED`

如果后端直接把整个 JSON 映射到内部对象，就容易出事。

------

## 4）它为什么和输入/输出控制都相关

因为它一方面是输入校验问题，另一方面也暴露了：

- 后端字段边界不清
- DTO / domain model 边界不清

------

## 5）对应哪些 rules

- **API8-R1**
- 也和 **API3-R4**（least privilege）有关系

------

# 九、CORS 是什么

这部分很多人会背定义，但不会放进系统图里理解。

------

## 1）最简单定义

**CORS（Cross-Origin Resource Sharing）** 是浏览器在跨域调用 API 时，服务端声明“哪些来源可以调我”的机制。

------

## 2）通俗理解

你可以把它理解成：

> “如果一个网页不是和 API 同域，它想调用这个 API，浏览器要先看服务端答不答应。”

------

## 3）为什么它只在浏览器场景特别重要

因为 CORS 是浏览器安全模型的一部分。
 不是所有客户端都有这个概念。

### 例如

- 浏览器前端：强相关
- 后端服务调后端服务：通常不靠 CORS
- batch job：通常不靠 CORS

------

## 4）最常见的几个控制点

### A. Allow Origin

哪些前端域名允许访问。

### B. Allow Methods

允许哪些 HTTP method。

### C. Allow Headers

允许哪些请求头。

### D. Credentials

是否允许跨域带凭证。

------

## 5）为什么 `Access-Control-Allow-Origin: *` 常被禁止

因为它相当于：

> “谁都可以从浏览器跨域来调我。”

这在很多企业 API 里太宽了，尤其涉及敏感数据时。

------

## 6）为什么 `Access-Control-Allow-Credentials` 也很敏感

如果允许跨域带 cookie/凭证，风险会明显增加。
 所以很多规则会对它非常保守。

------

## 7）你在 assessment 里怎么检查 CORS

看：

- gateway policy
- backend CORS config
- response header
- browser network response
- allowed origins 是否白名单
- 是否出现 wildcard

------

## 8）对应哪些 rules

- **API7-R1**
- **API3-R4**（avoid origin `*`）

------

# 十、Rate Limiting 是什么

------

## 1）最简单定义

**Rate Limiting** 就是：

> 限制某个调用方在单位时间内最多能发多少请求。

------

## 2）通俗理解

你可以把它理解成：

> “系统给每个调用方设了一个请求频率上限，防止有人狂刷。”

------

## 3）为什么它重要

因为很多风险不是“越权读数据”，而是：

- 暴力猜路径
- 猛刷高成本接口
- 打爆系统资源
- 枚举对象 ID
- 试探接口行为

所以 rate limit 一方面保护稳定性，另一方面也是安全控制。

------

## 4）按谁限最常见

企业里常见按：

- client_id
- user
- client + user
- IP
- 某类 endpoint

来限流。

------

## 5）为什么 gateway 特别适合做它

因为 gateway 是统一入口，很适合：

- 统一计数
- 统一拒绝
- 统一日志
- 统一 quota

------

## 6）在 assessment 里怎么检查

看：

- Apigee quota / spike arrest / rate limit policy
- 是否按 client/user/IP 配
- 高成本接口是否单独限制
- 限流触发后是否返回合理响应

------

## 7）对应哪些 rules

- **API4-R1**
- 也会帮助 **API7-R4**（防枚举）
- 和 **API8-R1**（防 brute force/参数攻击）有间接关系

------

# 十一、Error Handling 是什么

------

## 1）最简单定义

**Error Handling** 在这里指的是：

> 出错时，系统怎么对外返回错误信息。

------

## 2）为什么它是安全问题

因为很多系统出错时会不小心暴露：

- Java stack trace
- 内部类名
- SQL 细节
- 文件路径
- 版本信息
- 框架细节
- 下游服务地址

攻击者会非常喜欢这些信息。

------

## 3）通俗理解

你可以把它理解成：

> “系统出错时，应该只说用户该知道的话，不该把内部家底抖出来。”

------

## 4）一个错误做法

直接把异常堆栈原样返回给前端：

```
{
  "error": "java.lang.NullPointerException at com.xxx.service.CustomerService..."
}
```

这会暴露大量内部信息。

------

## 5）正确思路

对外：

- 返回标准 HTTP status
- 返回受控错误码 / 错误消息
- 不暴露内部实现细节

对内：

- 详细日志可以进日志系统
- 但日志也不能乱打敏感数据

------

## 6）你在 assessment 里怎么检查

看：

- gateway fault rule
- backend exception handler
- 浏览器 / Postman 实际错误响应
- 是否返回 stack trace
- 是否暴露 internal host/path/class name

------

## 7）对应哪些 rules

- **API7-R2**
- **API7-R3**（header 与错误行为相关）
- **API10-R1**（日志侧要区分“对外不暴露”与“对内可追踪”）

------

# 十二、Header 安全在这一章里怎么理解

前面 gateway 章讲过 header，这里从输入输出角度再补一下。

------

## 1）Content-Type

### 为什么重要

告诉客户端/浏览器怎么解释响应内容。
 如果不明确或错误，可能引发错误解析和安全问题。

------

## 2）X-Content-Type-Options: nosniff

### 通俗理解

告诉浏览器：

> 别自己猜内容类型，按我说的来。

### 为什么重要

可以减少某些内容被错误当成脚本执行的风险。

------

## 3）Cache-Control / Pragma / Expires

### 为什么重要

如果 API 响应里有敏感信息，不应该被随便缓存。

------

## 4）去掉 `Server`、`X-Powered-By`

### 为什么重要

减少攻击者获取技术栈信息。

------

## 5）对应哪些 rules

- **API7-R3**

------

# 十三、把这一章放回“Gateway 和 Backend 分工”

这一步非常关键。

------

## Gateway 更擅长做

- method/path/content-type 检查
- schema validation
- 基础参数格式检查
- rate limiting
- CORS
- 统一错误响应基线
- 一些安全 headers

------

## Backend 必须继续做

- 业务语义校验
- object ownership 校验
- field-level response trimming
- secret data 不暴露
- mass assignment 防护
- 细粒度业务错误控制

------

## 一句话记忆

- **Gateway**：先挡住“明显不对的请求”
- **Backend**：再决定“业务上到底行不行、能给多少”

------

# 十四、做 assessment 时，这一章怎么真正落地

以后你看到一个 API，可以固定按下面顺序问。

------

## 1）输入侧

- 有没有 schema / DTO 校验
- method / content-type 是否受控
- 参数类型/长度/格式是否受控
- 自由文本是否有清洗
- 文件上传是否有限制

------

## 2）输出侧

- 是否只返回必要字段
- 是否暴露 secret/sensitive data
- 是否存在 excessive data exposure
- 是否存在 mass assignment 边界问题

------

## 3）浏览器场景

- CORS 是否白名单
- 有没有 wildcard origin
- credentials 是否过宽

------

## 4）流量控制

- 有没有 rate limit / quota
- 是否按合适维度限制

------

## 5）错误响应

- 是否使用标准状态码
- 是否暴露 stack trace / internal details
- 是否对外克制、对内可追踪

------

# 十五、这一章和 rules 的对应关系

------

## 输入校验

- **API8-R1**：最核心
- 也间接支持 **API7-R4**

------

## 输出控制 / need-to-know / secret data

- **API3-R1**
- **API3-R2**
- **API1-R1**
- **API3-R4**

------

## CORS

- **API7-R1**
- **API3-R4**

------

## Rate Limiting

- **API4-R1**

------

## Error Handling

- **API7-R2**
- **API7-R3**

------

## 安全 headers

- **API7-R3**

------

# 十六、一个完整例子，把这一章串起来

假设前端调用一个“更新客户联系方式”的 API。

### 第一步：请求进 gateway

Gateway 检查：

- method 是不是 `PUT`
- content-type 是不是 `application/json`
- body 是否符合 schema
- rate limit 是否超了
- 来源 origin 是否允许

### 第二步：请求进后端

Backend 再检查：

- 这个 customerId 是不是当前用户可操作的
- 手机号格式是否业务上有效
- 某些字段是不是前端根本不该提交
- 是否有人偷偷加了 `isVip=true` 这种字段

### 第三步：后端返回响应

系统只返回前端需要看到的字段，不返回内部字段。

### 第四步：如果出错

对外只返回受控错误信息，不返回堆栈。

这就是一套完整的“输入到输出”的安全链。

------

# 十七、这一章最重要的 10 句话

## 第一句

**输入校验是在防“坏请求进来”。**

## 第二句

**输出控制是在防“不该给的东西出去”。**

## 第三句

**语法校验和语义校验不是一回事，前者 gateway 擅长，后者 backend 必须做。**

## 第四句

**Schema validation 是自动化做结构校验的很好方式。**

## 第五句

**前端不显示，不等于数据没有暴露。**

## 第六句

**CORS 是浏览器跨域访问控制，不是通用认证机制。**

## 第七句

**Rate limiting 既是稳定性控制，也是安全控制。**

## 第八句

**错误响应应该克制，不该把内部实现细节暴露给外部。**

## 第九句

**Header 安全是输出控制的一部分。**

## 第十句

**做 assessment 时，要同时看 gateway 的通用控制和 backend 的业务控制。**

------

## 十八、一句话总结这章

你以后看到一个 API，请固定问自己：
 **它收进来的东西有没有被严格检查，吐出去的东西有没有被严格收敛。**



# 第 14 部分：越权与暴露——BOLA、IDOR、technical API、predictable path

这一章你先抓住一句话：

> **很多高风险 API 漏洞，不是“你没登录”，而是“你登录了，但你看到了不该看的东西”。**

这就是为什么你前面 rules 里会反复强调：

- fine-grained authorization
- need-to-know
- least privilege
- technical API not exposed
- predictable resource location
- admin vs user endpoint separation

------

## 一、先建立总图：这一章其实在看 4 类问题

以后你做 assessment，看到“暴露面”或“越权”，可以先分成这 4 类：

### 1. 对象级越权

你本来只能看对象 A，却能通过改 ID 看到对象 B。

### 2. 字段级 / 数据级暴露

你本来不该看到某些字段，但系统还是返回了。

### 3. 技术接口暴露

本来只该内部用的技术接口、管理接口，被放到了外网或过大暴露面。

### 4. 路径和资源太可猜

攻击者很容易猜到：

- 经典 API 路径
- 隐藏接口
- 可枚举对象路径
- 管理型 URL

------

# 二、先讲“越权”到底是什么

------

## 1）最简单定义

**越权** 就是：

> 请求方做到了本来不该被允许的事情。

### 通俗理解

系统没有正确限制“你能做什么”，于是你做了超出自己权限边界的事情。

------

## 2）越权不一定是“管理员权限”

很多新人一听“越权”就想到：

- 普通用户变管理员

那只是其中一种。

更常见的是：

- 你本来只能看自己的账户，却看到了别人的账户
- 你本来只能看摘要字段，却看到了内部字段
- 你本来只能访问用户接口，却访问了管理接口

------

## 3）为什么越权特别危险

因为这类问题的特点是：

- 登录可能是合法的
- token 可能是真的
- gateway 也可能放行了
- 但系统最终还是把不该给你的东西给了你

所以它很容易绕过“表面合规感”。

------

# 三、BOLA 是什么

这是 API 安全里最关键的概念之一。

------

## 1）最简单定义

**BOLA = Broken Object-Level Authorization**
 意思是：

> 对“具体业务对象”的授权检查坏掉了。

------

## 2）通俗理解

你可以把它理解成：

> 系统知道你能访问“这类 API”，但没有继续检查“你能不能访问这个具体对象”。

------

## 3）最经典的例子

用户原本请求自己的账户：

```
GET /accounts/123
```

然后他把 URL 改成：

```
GET /accounts/124
```

如果系统只验证：

- 你已登录
- 你有 `read_accounts` scope

但没有验证：

- `124` 是不是你的账户

那就是典型 BOLA。

------

## 4）为什么 BOLA 在企业里特别常见

因为很多系统确实做了这些：

- 登录
- token
- role
- scope

但没有做最后一步：

- **对象 ownership / object-level authorization**

而 API 又天然大量使用对象 ID：

- accountId
- customerId
- contractId
- orderId

所以 BOLA 特别常见。

------

## 5）你在 assessment 里怎么判断是否有 BOLA 风险

你先问 3 个问题：

### A. 请求里有没有用户可控的对象标识

比如：

- path parameter
- query parameter
- body 里的 object id

### B. 后端有没有重新检查这个对象与当前 user/client 的关系

比如：

- 这个 accountId 是不是属于当前用户
- 这个 customerId 是否属于当前 tenant
- 这个 contractId 是否在当前员工权限范围内

### C. 系统是不是只是“查得到就返回”

如果代码像这样：

```
repository.findById(id)
return entity;
```

而没有 ownership check，那风险就很大。

------

## 6）BOLA 和哪些 rules 关系最强

- **API1-R1**
- **API3-R1**
- **API3-R4**
- 也和 **API7-R4** 间接相关（因为路径/对象可枚举会放大风险）

------

# 四、IDOR 是什么

它和 BOLA 很像，经常会一起出现。

------

## 1）最简单定义

**IDOR = Insecure Direct Object References**
 意思是：

> 系统直接把内部对象标识暴露给客户端使用，但没有做好授权保护。

------

## 2）通俗理解

你可以把它理解成：

> “系统把门牌号直接给了你，但没有确保你只能进你自己的房间。”

------

## 3）和 BOLA 的关系怎么记

你现在先不用特别纠结教科书级区分，先这样记就够了：

- **IDOR** 更强调：对象引用方式本身很直接、很可猜
- **BOLA** 更强调：对象级授权检查没有做好

在实际 assessment 里，这两个经常一起分析。

------

## 4）一个最简单例子

URL 里直接暴露：

```
GET /customers/100245
```

如果：

- 这个 ID 很容易枚举
- 后端又没做授权校验

那既有 IDOR 风味，也有 BOLA 风险。

------

## 5）在 assessment 里怎么用

你看到：

- 连续可猜的 ID
- path 里直接暴露 object id
- query parameter 里直接传 object id

时，不要立刻说“有漏洞”，但要马上追问：

> **后端有没有对这个 object id 做授权校验。**

如果有校验，风险会小很多。
 如果没有校验，就很危险。

------

## 6）和哪些 rules 关系强

- **API1-R1**
- **API3-R1**
- **API7-R4**

------

# 五、对象级授权（Object-level Authorization）是什么

这一节其实是在告诉你：BOLA 的“正确解法”是什么。

------

## 1）最简单定义

**对象级授权** 就是：

> 系统不仅检查你能不能访问这个 API，还检查你能不能访问这个具体对象。

------

## 2）通俗理解

你可以把它理解成：

> “不是你有钥匙就能进所有房间，而是要确认你这把钥匙能不能开这扇具体的门。”

------

## 3）为什么它通常必须在 backend 做

因为 backend 更容易知道：

- 当前 user 是谁
- 这个 object 属于谁
- 这个对象是什么状态
- 当前上下文允许什么动作

Gateway 往往只知道：

- 这个 token 有某个 scope

但不知道：

- `accountId=124` 属于谁

------

## 4）在项目代码里长什么样

你理想中会看到类似这样的逻辑：

- 根据 token/session 拿到当前 user identity
- 根据 request 取出 object id
- 查对象归属/权限关系
- 不符合则拒绝

不是只做：

- `if (hasRole("USER"))`

就完了。

------

## 5）和哪些 rules 关系最强

- **API1-R1**
- **API3-R1**
- **API3-R4**

------

# 六、字段级授权（Field-level Authorization）是什么

这个是比对象级再细一层的控制。

------

## 1）最简单定义

**字段级授权** 就是：

> 同一个对象可以返回，但不是所有字段都对所有请求方可见。

------

## 2）通俗理解

你可以把它理解成：

> “你可以进这个房间，但不代表你能打开房间里所有抽屉。”

------

## 3）银行场景例子

一个 customer profile 对象里可能有：

- name
- email
- phone
- riskScore
- internalStatus
- operatorNotes

客户自己可能只应看到：

- name
- email
- phone

而不该看到：

- riskScore
- internalStatus
- operatorNotes

------

## 4）为什么这也属于越权的一部分

因为即使对象本身允许访问，字段也可能不该全给。

这就是：

- **对象可见 ≠ 字段全可见**

------

## 5）在 assessment 里怎么检查

你要看：

- response DTO 是否专门裁剪过
- 不同角色/不同 client 是否返回不同字段
- backend 是否只返回前端需要的数据
- 有没有“整个 entity 直接序列化返回”的危险做法

------

## 6）和哪些 rules 关系强

- **API3-R1**
- **API3-R2**
- **API1-R1**
- **API3-R4**

------

# 七、Admin vs User Endpoints：管理接口和普通接口为什么必须分开

------

## 1）最简单理解

不是所有 API 都是给同一类人用的。

有些是：

- 普通用户接口

有些是：

- 管理接口
- 运维接口
- 审批接口

------

## 2）为什么必须分开

因为如果 admin 接口和 user 接口边界不清，会导致：

- 普通用户误打到管理接口
- 文档和路径暴露太多
- 授权策略混乱
- 攻击者更容易试探高权限端点

------

## 3）理想状态

你希望系统至少做到：

- 路径/产品/暴露面分开
- 认证授权策略分开
- 文档/前端入口分开
- 对外不暴露不该暴露的 admin/technical path

------

## 4）和哪些 rules 关系强

- **API1-R1**
- **API3-R3**
- **API7-R4**
- 也和 **API3-R4** 有关系

------

# 八、Technical API 是什么

这个是企业里特别重要、但很多新人不敏感的概念。

------

## 1）最简单定义

**Technical API** 指的是：

> 不是给业务用户正常使用的，而是给技术基础设施、管理功能、运维或平台用途使用的 API。

------

## 2）通俗理解

你可以把它理解成：

> “系统后厨和机房用的接口，不是大厅客户该碰到的接口。”

------

## 3）常见例子

- 日志查看接口
- 技术库存接口
- CI/CD 管理接口
- Kubernetes 管理接口
- 配置管理接口
- CMS / 管理后台 API
- Actuator / admin endpoints（有些场景）

------

## 4）为什么它危险

因为 technical API 往往：

- 权限高
- 功能强
- 数据敏感
- 给内部人员或系统用
- 一旦暴露，攻击面巨大

------

## 5）为什么企业 rule 会明确说它不能对互联网暴露

因为这类接口不是普通对外业务 API。
 就算有鉴权，它也通常不该出现在互联网暴露面上。

------

## 6）你在 assessment 里怎么查

你要看：

- Swagger / OpenAPI 里有没有管理型路径
- gateway route 是否暴露技术路径
- actuator / admin endpoint 是否对外可达
- ingress / LB 是否让技术接口上公网
- 是否混在普通业务 API 产品里

------

## 7）和哪些 rules 关系最强

- **API3-R3**
- **API7-R4**
- **API0-R3**

------

# 九、Predictable Resource Location 是什么

这个概念在你前面的 Protection rules 里单独出现了。

------

## 1）最简单定义

**Predictable Resource Location** 指的是：

> 攻击者太容易猜到你的 API 路径、资源位置、隐藏功能或技术接口。

------

## 2）通俗理解

你可以把它理解成：

> “门藏得太显眼，别人一猜就猜到。”

------

## 3）为什么这会构成风险

因为攻击者即使不知道完整文档，也可以靠枚举和猜测发现：

- `/api`
- `/api/v1`
- `/admin`
- `/swagger`
- `/actuator`
- `/apis.json`
- `/internal`
- `/test`

这些经典路径一旦存在，又保护不严，就容易出事。

------

## 4）它不等于“路径不能有规律”

你不要走到另一个极端，以为所有 API 都要故意起怪名字。

问题不在于“规范命名”，而在于：

- 不该对外暴露的路径是不是很容易被猜到
- 内部技术功能是不是暴露在了外部可访问面上
- 枚举之后是否能得到有价值反馈

------

## 5）和对象 ID 枚举的关系

这部分也会和 object-level 风险结合：

- `/customers/1`
- `/customers/2`
- `/customers/3`

如果对象 ID 连续可猜，后端又没做好授权，风险会被放大。

所以 predictable location 常常不是单独致命，而是和：

- BOLA
- IDOR
- 弱错误处理
- 无 rate limiting

一起叠加变危险。

------

## 6）你在 assessment 里怎么查

你要看：

### A. 外部暴露路径

- 是否存在管理型路径
- 是否存在经典隐藏接口路径

### B. 文档与 crawlability

- 搜索引擎是否可索引
- Swagger 是否无控制暴露
- robots / indexing 策略是否缺失

### C. 枚举反馈

- 猜路径时是否返回太多信息
- 不存在和存在路径的响应差异是否过于明显

------

## 7）和哪些 rules 关系最强

- **API7-R4**
- **API3-R3**
- **API0-R3**
- 也和 **API4-R1**、**API7-R2** 有间接关系

------

# 十、为什么“可猜 + 无限试 + 错误信息太多”会叠加成大问题

这一点非常企业实践化。

一个系统如果同时存在：

- 路径可猜
- object id 可枚举
- 没有限流
- 错误信息很详细
- technical API 也暴露着

那攻击者就会很舒服地做：

- 枚举路径
- 枚举对象
- 观察错误差异
- 逐步摸清系统边界

所以你要开始习惯：

> **不要孤立看一条规则，要看几条规则叠在一起后的风险。**

### 这里常见叠加关系

- **API7-R4**（路径可猜）
- **API4-R1**（没限流）
- **API7-R2**（错误太详细）
- **API1-R1**（对象级授权不严）
- **API3-R3**（技术接口暴露）

------

# 十一、这一章在代码和系统里怎么落地查

这部分最实用。

------

## 1）看 URL / Controller / Route 设计

找：

- path 里有没有直接暴露 object id
- admin 与 user 路径是否分开
- technical endpoint 是否单独隔离
- 是否有典型暴露路径

------

## 2）看 Swagger / OpenAPI

看：

- 对外文档里有哪些接口
- 是否有明显管理接口
- 是否把 technical API 也放进外部产品
- 返回 schema 是否给太多字段

------

## 3）看后端代码

看：

- 根据 object id 查数据后，是否再校验 ownership
- 返回 DTO 是否裁剪字段
- 是否整个 entity 直接返回
- 是否角色校验后就草率放行，没有对象级判断

------

## 4）看网关

看：

- 外部暴露了哪些 path
- 是否有路径级分层
- 是否对管理路径额外保护
- 是否存在不该暴露的 route

------

## 5）看实际响应

很重要。
 对几个不同 ID 和路径尝试时，看：

- 返回 403/404/200 的差异
- 错误信息是否泄露对象存在性
- 是否容易通过响应差异枚举资源

------

# 十二、这一章和 rules 的对应关系

------

## 对象级授权 / BOLA / IDOR

- **API1-R1**
- **API3-R1**
- **API3-R4**

------

## 字段级暴露 / need-to-know / secret data

- **API3-R1**
- **API3-R2**
- **API1-R1**

------

## Technical API 暴露

- **API3-R3**

------

## Predictable path / 可枚举资源位置

- **API7-R4**

------

## 叠加辅助规则

- **API4-R1**（防枚举/滥刷）
- **API7-R2**（避免给攻击者太多反馈）
- **API0-R3**（统一通过 gateway 暴露）

------

# 十三、一个完整例子，把这一章串起来

假设一个客户服务 API 有这些接口：

- `GET /customers/{customerId}`
- `GET /admin/customers/{customerId}`
- `GET /actuator/health`
- `GET /api/v1/customers`

### 你做 assessment 时怎么想

#### 1. 对象级风险

普通用户如果把 `{customerId}` 改掉，会不会看到别人客户资料？

#### 2. 字段级风险

即使能看自己的 customer profile，是否顺便拿到了内部字段？

#### 3. admin 路径风险

普通用户或外部调用方是否能摸到 `/admin/...`？

#### 4. technical API 风险

`/actuator/health` 是否被对外暴露？

#### 5. predictable path 风险

路径太经典，攻击者是不是很容易枚举和试探？

#### 6. 如果再加上没限流、错误太详细

那整体风险会明显上升。

------

# 十四、这一章最重要的 10 句话

## 第一句

**越权不一定是“变管理员”，更常见的是“看到了不属于自己的对象或字段”。**

## 第二句

**BOLA 的本质是对象级授权没做好。**

## 第三句

**IDOR 强调对象引用太直接、太可猜，而授权保护又不够。**

## 第四句

**有 API scope，不等于就能访问任意对象。**

## 第五句

**对象级授权通常必须在 backend 做，gateway 不足以单独解决。**

## 第六句

**字段级授权是 need-to-know 在响应层的具体体现。**

## 第七句

**Technical API 不应该暴露在外网暴露面上。**

## 第八句

**Predictable path 本身未必致命，但会放大枚举、越权和侦察风险。**

## 第九句

**路径可猜、限流缺失、错误太详细、对象授权薄弱，组合起来会很危险。**

## 第十句

**做 assessment 时，要同时看“能不能进 API”和“进来后能不能碰到不该碰的对象、字段和技术接口”。**

------

## 十五、一句话总结这章

你以后看到任何 API 路径、对象 ID、管理接口时，都先问自己：
 **这个调用方会不会通过改 ID、猜路径、碰到技术接口，拿到本来不该拿到的东西。**



# 第 15 部分：日志、监控、追踪、SIEM——出了问题能不能看到

这一章你先抓住一句话：

> **没有可追踪性，很多安全控制就算做了，也很难证明它生效，更难在出事后定位问题。**

------

## 一、先说“可追踪性”到底是什么

### 1）最简单定义

**Traceability（可追踪性）** 就是：

> 系统发生某个安全相关行为后，你能不能事后知道“谁、什么时候、从哪里、做了什么、结果怎样”。

------

### 2）通俗理解

你可以把它理解成：

> “系统有没有留下足够清楚的行踪记录。”

不是为了“监视用户”，而是为了：

- 发现异常
- 追查问题
- 定位责任边界
- 还原攻击路径
- 满足审计要求

------

### 3）为什么企业里特别重视它

因为在真实环境里，很多安全事件并不是你当场看见的，而是：

- 过了几小时才发现异常
- 过了几天才发现数据被访问
- 过了几周审计时才发现有问题

这时如果没有日志和链路追踪，基本就很难判断：

- 谁发起的
- 走了哪条链路
- gateway 放行了什么
- backend 返回了什么
- 哪一层失守了

------

# 二、Logging（日志）是什么

------

## 1）最简单定义

**Logging** 就是系统把关键事件记录下来。

### 通俗理解

你可以把它理解成：

> “系统在写运行日记。”

------

## 2）为什么日志不是“越多越好”

很多新人第一反应是：

- 多打点日志总没错

不对。日志要解决两个矛盾：

### 矛盾 A：要足够多，才能追查问题

### 矛盾 B：不能乱打，否则会泄露敏感信息

所以企业日志不是“多”，而是：

> **记录关键上下文，但不要把不该记录的东西写进去。**

------

## 3）安全 assessment 里最关心哪些日志

你可以先把安全相关日志分成 4 类：

### A. 身份与登录相关

- 登录成功/失败
- token 校验失败
- access denied
- 权限不足
- MFA 失败

### B. 访问控制相关

- 越权尝试
- scope 不足
- 角色不匹配
- object ownership 校验失败

### C. 输入与攻击相关

- 非法参数
- schema validation 失败
- 恶意 payload
- rate limit 命中
- CSRF / state / nonce 校验失败

### D. 敏感操作相关

- 数据修改
- 权限变更
- 同意授权/撤销
- 管理操作
- 高风险业务动作

------

# 三、为什么“该记什么”很重要

你前面 rules 里其实已经给过方向了。
 不是所有东西都值得记，但有一些事件必须记。

------

## 1）登录尝试

为什么必须记：

- 可以发现爆破
- 可以发现异常登录来源
- 可以做安全审计

------

## 2）访问控制失败

为什么必须记：

- 这是最直接的攻击和误用信号之一
- 可以帮助发现 BOLA/IDOR 试探
- 可以发现 client 权限配置异常

------

## 3）敏感数据变更和敏感动作

为什么必须记：

- 后续审计要知道谁改了什么
- 问题追责要有证据
- 异常业务动作要能回溯

------

## 4）输入校验失败

为什么必须记：

- 可能代表攻击测试
- 可能代表脚本扫接口
- 可能代表 payload 注入尝试

------

## 5）Consent / token / security events

为什么必须记：

- token 撤销
- consent change
- refresh token 异常
- 权限变化
   这些都和身份安全紧密相关。

------

# 四、日志里应该记哪些上下文

这一块很重要。
 不是只写一句“error happened”就算日志。

------

## 1）Who：是谁

可能包括：

- user id
- client_id
- subject
- service account
- source system

### 为什么重要

因为后面你要回答：

- 到底是哪个用户
- 哪个 app
- 哪个服务

------

## 2）When：什么时候

至少要有：

- 时间戳
- 时区一致性
- 必要时精确到毫秒

### 为什么重要

因为系统是分布式的，没有准确时间很难串联事件。

------

## 3）Where from：从哪里来

可能包括：

- source IP
- origin
- user agent（视场景）
- gateway / service name
- environment

### 为什么重要

可以帮助你识别：

- 异常来源
- 攻击路径
- 内外网边界

------

## 4）What：做了什么

要能看出：

- 请求了哪个 API
- 哪个 method
- 什么操作类型
- 是否命中某个安全策略
- 最终结果是成功还是失败

------

## 5）Result：结果如何

比如：

- success
- denied
- validation_failed
- rate_limited
- unauthorized
- forbidden

------

## 6）Why：为什么失败

这里要注意平衡。

### 对外响应

不能太详细，防止信息泄露。

### 内部日志

可以记录更具体的失败原因，但仍不能乱打敏感数据。

------

# 五、Correlation ID / Request ID / Trace ID：为什么一定要串起来

你前面已经学过 `correlation id`，这一章把它放回日志和追踪里。

------

## 1）为什么光有日志还不够

因为在企业系统里，一次请求常常会经过：

- 前端
- gateway
- BFF
- service A
- service B
- DB / downstream

如果每层都各写各的日志，但没有共同标识，你很难知道：

- 这些日志是不是同一次请求的

------

## 2）Correlation ID 是干什么的

它的核心作用就是：

> 把同一次调用链上的日志串起来。

------

## 3）通俗理解

你可以把它理解成：

> “给这一次完整请求流程贴一张统一工单号。”

然后：

- gateway 记这个号
- backend 也记这个号
- 下游服务也记这个号

这样一查就能串起来。

------

## 4）它在安全上为什么特别有价值

因为你后面分析安全事件时，经常要回答：

- 这次越权尝试从哪开始
- gateway 有没有先拦
- backend 为什么返回了 403/404/200
- 哪个服务先出错
- 攻击者有没有反复试探多个路径

没有 correlation id，这些会非常难查。

------

## 5）在 assessment 里怎么问

你要看：

- 请求进入 gateway 时是否生成/透传 correlation id
- 后端日志是否带同一个 id
- 分布式追踪/监控平台是否能按这个 id 查整条链

------

# 六、Monitoring（监控）是什么

------

## 1）最简单定义

**Monitoring** 就是持续观察系统运行状态和关键安全/稳定性指标。

### 通俗理解

你可以把它理解成：

> “系统有没有实时体检和看门人。”

------

## 2）日志和监控的区别

### 日志

更像：

- 详细事件记录
- 事后追查材料

### 监控

更像：

- 实时看整体状态
- 指标和趋势
- 异常告警基础

------

## 3）安全相关监控常看什么

比如：

- 401/403 突然暴增
- 某 client_id 调用量异常
- 某 IP 高频失败
- 某路径被大量试探
- rate limit 命中暴增
- token validation fail 暴增
- unusual geographic access（视系统而定）

------

## 4）为什么企业需要监控而不只是日志

因为纯日志是“有记录”，
 但监控才更接近：

> **能及时发现问题。**

------

# 七、Alerting（告警）是什么

------

## 1）最简单定义

**Alerting** 就是当某类异常达到阈值或模式时，主动通知人或系统。

### 通俗理解

你可以把它理解成：

> “不是等人来查日志，而是系统先响铃。”

------

## 2）为什么告警重要

因为很多安全问题不能靠人工慢慢翻日志发现，必须尽量早发现。

------

## 3）安全里常见的告警场景

例如：

- 登录失败暴增
- 某 API 403/401 激增
- 某 IP 快速枚举资源
- rate limit 命中异常
- 管理接口被访问
- token 校验失败异常增加
- 敏感数据变更异常频繁

------

# 八、SIEM 是什么

这是企业安全场景里非常重要的概念。

------

## 1）最简单定义

**SIEM（Security Information and Event Management）** 是用来集中收集、关联分析和告警安全事件的平台。

### 通俗理解

你可以把它理解成：

> “安全版的中央监控与情报中枢。”

------

## 2）为什么企业需要 SIEM

因为日志散在各系统里不够用。
 企业需要把：

- gateway 日志
- backend 日志
- 认证系统日志
- WAF / firewall / IDS / IPS 日志
- 主机日志
- 云平台日志

集中起来，做统一分析。

------

## 3）SIEM 能做什么

### A. 集中汇总

不再每个系统单独翻日志。

### B. 关联分析

比如发现：

- 某 IP 在 gateway 试探路径
- 同时某用户出现异常登录失败
- 后端又出现大量 403

这些单看不明显，但合起来可能就是攻击迹象。

### C. 告警

基于规则或行为模式触发告警。

### D. 审计

满足企业合规和事后追查要求。

------

## 4）在 assessment 里怎么理解

你不一定要会操作 SIEM，但你要知道：

- 项目日志是不是被集中接入
- 是否能支撑安全事件关联分析
- 是否只是“本地打印日志”而已

------

## 5）相关 rules

- **API10-R1**
- **API7-R6**
- **API7-R7**

------

# 九、Traceability（可追踪性）到底和普通 logging 有什么区别

------

## 1）普通 logging

更偏：

- 某个服务自己记了什么

## 2）Traceability

更偏：

- 从全局看，能不能把一整条安全事件链条串起来

所以 Traceability 通常要求：

- 有足够事件
- 有足够上下文
- 有 correlation id
- 有集中化
- 有留存
- 有防篡改

------

## 3）你可以这样记

### Logging

是“有日记”

### Traceability

是“日记能串起来，事后能查清楚”

------

# 十、为什么“日志不能留敏感数据”这么重要

这一点一定要认真理解，因为很多人觉得“多打点方便排查”。

------

## 1）如果日志里有 token / password / card number 会怎样

那日志本身就变成了高价值攻击目标。

攻击者甚至不一定去打业务逻辑，直接想办法看日志就够了。

------

## 2）哪些东西尤其不该进日志

你先记最核心的一批：

- password
- raw access token
- refresh token
- private key material
- full card number
- 高敏感 personal data（除非有明确必要且合规）

------

## 3）为什么“必要排查信息”和“敏感数据”要平衡

你希望日志能帮助定位问题，但不能为了方便定位就把家底都写进去。

所以企业实践通常是：

- 记录必要上下文
- 对敏感数据做 masking/redaction
- 对不该记录的内容完全不记录

------

## 4）在 assessment 里怎么查

你可以看：

- 代码里有没有 `log.info("token={}", token)` 这种危险写法
- 异常堆栈是否可能连带打印敏感 payload
- 网关日志是否记录 Authorization header
- 监控平台是否收集了敏感字段

------

## 5）相关 rules

- **API10-R1**
- **API2-R5**
- **API7-R2**

------

# 十一、日志防篡改和集中化，为什么企业规则会特别强调

------

## 1）为什么“本地日志”不够

如果日志只留在本机：

- 攻击者拿下主机后可以删日志
- 容器重建可能丢日志
- 排查时分散、难查
- 不利于集中告警

------

## 2）为什么要集中化

把日志实时汇总到中央平台后：

- 更难被单点篡改
- 更容易做统一检索
- 更容易做安全分析
- 更方便保留足够长时间

------

## 3）为什么要防篡改

因为如果日志能被随便改，攻击者做完事后删痕迹，你就失去证据了。

------

## 4）相关 rules

- **API10-R1**

------

# 十二、日志留存（Retention）为什么是安全要求的一部分

------

## 1）为什么不能只保留几天

因为很多安全事件不是当天就发现的。

可能：

- 几周后审计才发现
- 几个月后排查才回溯到某次异常

------

## 2）为什么规则会写“至少保存 12 个月”

这就是典型的企业审计思维：

- 有足够长的可追溯期
- 能支持审计和事后调查

------

## 3）你在 assessment 里怎么理解

不是只看“能不能打印日志”，还要看：

- 是否有 retention policy
- 是否集中保存
- 是否满足企业最低留存要求

------

# 十三、这一章在项目里怎么真正检查

这部分最实用。

------

## 1）看代码

你可以看：

- login / auth 逻辑有没有记录成功/失败
- access denied 是否有日志
- validation failure 是否有日志
- 敏感操作是否有 audit log
- 有没有直接打印 token / password / 敏感 payload

------

## 2）看 gateway / Apigee

看：

- 是否记录请求路径、状态码、client_id、来源 IP
- 是否记录 rate limit 触发
- 是否记录 token 验证失败
- 是否透传 correlation id

------

## 3）看 backend

看：

- 日志中是否带 request/correlation id
- 是否有统一异常处理和审计日志
- 是否对高风险操作打 audit log

------

## 4）看平台与运维

看：

- 日志是否进中央平台
- 是否接 SIEM
- 是否有 retention policy
- 是否有异常告警

------

## 5）看实际现象

如果你能操作环境，可以实际验证：

- 登录失败后有没有日志
- 越权请求后有没有日志
- rate limit 命中后有没有日志
- correlation id 是否贯穿

------

# 十四、这一章和 rules 的对应关系

------

## API10-R1 — Traceability

这是本章最核心、最直接的 rule。

重点看：

- 记什么事件
- 是否有上下文
- 是否集中化
- 是否防篡改
- 是否长期留存
- 是否不泄露敏感信息

------

## API7-R6 — Vulnerability testing

和本章不是直接一回事，但测试与监控经常联动。
 测试时需要看：

- 日志能否支持发现问题
- 监控能否发现异常行为

------

## API7-R7 — Penetration testing

同理，渗透测试后你往往会观察：

- 是否留下足够痕迹
- 告警是否被触发
- traceability 是否有效

------

## API2-R5 — Token management

因为 token 不该被打入日志，所以和本章也强相关。

------

# 十五、一个完整例子，把这一章串起来

假设一个攻击者在试探客户 API。

### 第一步

他不断改 `customerId` 试探对象是否存在。

### 第二步

Gateway 记录：

- 来源 IP
- client_id
- path
- 状态码
- correlation id

### 第三步

Backend 记录：

- 当前 user identity
- object ownership 校验失败
- access denied
- 同一个 correlation id

### 第四步

集中日志平台 / SIEM 发现：

- 某个 IP 短时间大量 403/404
- 同一类路径被快速枚举
- 命中 rate limit

### 第五步

安全团队收到告警，回看整条链路，定位这次行为的来源和影响范围。

这就是“日志、追踪、监控、告警”一起工作时的样子。

------

# 十六、这一章最重要的 10 句话

## 第一句

**日志不是越多越好，而是要记关键事件和关键上下文。**

## 第二句

**安全日志最重要的是：谁、什么时候、从哪里、做了什么、结果如何。**

## 第三句

**有日志不等于可追踪，可追踪要求能把整条链串起来。**

## 第四句

**Correlation ID 是把 gateway、backend、下游日志串成一条链的关键。**

## 第五句

**监控更偏实时观察，日志更偏事后追查。**

## 第六句

**SIEM 是企业用来集中分析安全事件的中枢，不只是存日志。**

## 第七句

**日志里不能乱放 token、密码、卡号等敏感数据。**

## 第八句

**日志应尽量集中化，并防篡改。**

## 第九句

**留存时间不足，会让很多安全事件失去可追溯性。**

## 第十句

**做 assessment 时，不要只问“有没有日志”，要问“能不能发现、能不能串联、能不能安全保存”。**

------

## 十七、一句话总结这章

你以后看到任何安全控制时，都要再补问一句：
 **如果有人绕过、试探、误用或者攻击了这层控制，系统能不能看见，并且事后查清楚。**



# 第 16 部分：安全测试与治理——SAST、DAST、pentest、inventory、ownership

这一章你先抓住一句话：

> **安全不仅要“做出来”，还要“被验证、被管理、被负责”。**

------

## 一、先建立总图：这一章其实在看 5 件事

以后你做 assessment，碰到治理和测试类 rule，可以先分成这 5 类：

### 1. 代码有没有做安全检查

也就是：

- SAST
- code audit
- 代码层面漏洞检查

### 2. 运行起来后有没有做安全测试

也就是：

- DAST
- vulnerability testing
- API fuzzing
- pentest

### 3. API 有没有被盘点和登记

也就是：

- inventory
- registry
- 文档、owner、分类、scope 是否明确

### 4. 有没有明确责任人

也就是：

- IT owner
- Business owner
- 谁负责下线、整改、审批

### 5. 生命周期有没有闭环

也就是：

- major release 前测不测
- critical/high 漏洞修不修
- 不用的 API 会不会下线

你可以把这一章理解成：

> **安全工作的“组织化和可交付化”。**

------

# 二、SAST 是什么

------

## 1）最简单定义

**SAST = Static Application Security Testing**
 也就是：

> **不把程序跑起来，直接分析代码、配置、依赖，找潜在安全问题。**

------

## 2）通俗理解

你可以把它理解成：

> “像代码审计机器人，在源码层面先帮你找一批危险写法。”

------

## 3）它能发现什么类型的问题

典型包括：

- SQL injection 风险
- command injection 风险
- 路径穿越
- 不安全反序列化
- 弱加密用法
- 不安全日志输出
- 明文敏感信息
- 某些 unsafe API 调用
- 配置错误
- 代码里的明显漏洞模式

------

## 4）它不能完全解决什么

SAST 很有用，但不是万能的。

它通常不擅长完整判断：

- 真实运行时行为
- 复杂业务授权问题
- BOLA/IDOR 是否真的成立
- 网关是否实际配置生效
- 某个接口从外部是否真的能到达

所以你不能把 SAST 当成“全部安全”。

------

## 5）为什么企业一定喜欢它

因为它适合：

- 集成到 CI/CD
- 尽早发现问题
- 在上线前就暴露一批风险
- 对代码层控制形成基线

------

## 6）你在工作里可能见到什么工具

你前面已经接触过一些，例如：

- Fortify
- SonarQube（严格说不完全等于安全工具，但有部分安全规则）
- Checkmarx（通用例子）
- 其他 SAST 平台

------

## 7）在 assessment 里怎么用

你要问：

- 项目是否做了 SAST
- 在什么阶段做（开发时、CI 时、上线前）
- critical/high findings 是否被修复
- 有没有只是“跑了工具”，但结果没人管

------

## 8）主要涉及哪些 rules

- **API8-R2**：最直接
- 也会支持很多别的 rule 的证据，但不是替代品

------

# 三、DAST 是什么

------

## 1）最简单定义

**DAST = Dynamic Application Security Testing**
 也就是：

> **把应用跑起来，从外部像攻击者一样去测试它。**

------

## 2）通俗理解

你可以把它理解成：

> “不是看代码，而是从接口外面实际试它会不会露问题。”

------

## 3）它能发现什么

更适合发现：

- 实际暴露的接口问题
- 某些输入校验缺陷
- 错误响应暴露
- header 问题
- CORS 问题
- 某些认证/授权缺陷
- 运行时配置问题

------

## 4）它和 SAST 的区别

这个一定要分开。

### SAST

更像：

- 看源代码
- 在“没跑起来”时检查

### DAST

更像：

- 从外部实际打接口
- 看系统运行起来后的表现

------

## 5）为什么两者都需要

因为它们看的角度不同：

- SAST 看到的是“代码可能有问题”
- DAST 看到的是“系统现在真的暴露了什么问题”

------

## 6）在 assessment 里怎么用

你要问：

- 项目有没有做动态安全测试
- 是自动化还是人工结合
- 是否覆盖 internet-facing API
- 发现的问题有没有闭环整改

------

## 7）主要涉及哪些 rules

- **API7-R6**
- 也间接支持：
  - API7-R2
  - API7-R3
  - API8-R1
  - API7-R4

------

# 四、Pentest（渗透测试）是什么

------

## 1）最简单定义

**Penetration Testing** 是：

> 由安全测试人员以更接近真实攻击者的方式，针对系统做更深入的人工/半人工攻击测试。

------

## 2）通俗理解

你可以把它理解成：

> “不是只跑自动化工具，而是真人带着攻击思路去试系统哪里能被打穿。”

------

## 3）它和 DAST 的区别

这两个很多新人会混。

### DAST

更偏自动化、广覆盖、运行时检测。

### Pentest

更偏人工深入、攻击链思维、重点突破。

------

## 4）为什么 pentest 特别重要

因为很多高风险问题，自动化工具不一定能真正判断，比如：

- 复杂授权绕过
- BOLA / IDOR
- 管理接口暴露
- 业务逻辑漏洞
- 多步流程漏洞
- 组合型攻击路径

------

## 5）什么叫 gray-box

你前面 rule 提到 “gray-box type of test”。

### 最简单理解

测试者不是完全黑盒啥都不知道，也不是拥有全部内部代码和最高权限，
 而是知道一部分系统信息，例如：

- 架构概览
- API 文档
- 部分账号
- 测试环境说明

这比纯黑盒更贴近企业真实测试。

------

## 6）在 assessment 里怎么用

你要问：

- 是否做了 pentest
- 是 major release 前做，还是只是偶尔做
- 测试范围是否包含互联网暴露 API
- critical/high 是否上线前修复
- 报告里是否有未闭环风险

------

## 7）主要涉及哪些 rules

- **API7-R7**：最直接
- 也间接支持很多运行时安全判断

------

# 五、Vulnerability Testing 和 Pentest 有什么区别

这个在 rules 里是分开的，你要会区分。

------

## 1）Vulnerability Testing

更偏：

> 系统性检查漏洞和弱点有没有存在。

通常会更流程化、频率化、覆盖广。

------

## 2）Pentest

更偏：

> 用攻击者思路深入验证系统能不能被打穿。

通常更人工、更有攻击链、更聚焦高风险点。

------

## 3）你先怎么记

- **Vulnerability testing**：更像体检和扫描
- **Pentest**：更像实战攻防演练

------

## 4）为什么企业 rules 要分开写

因为两者不是替代关系，而是互补：

- 前者偏持续性和覆盖面
- 后者偏深度和攻击真实性

------

# 六、Code Audit 是什么

这个和 SAST 很接近，但不完全等于。

------

## 1）最简单理解

**Code Audit** 就是对代码做安全审查。

### 可能包括

- 工具扫描
- 人工 review
- 针对高风险模块做深入检查

------

## 2）为什么 rule 会写“A code audit must be performed”

因为企业想表达的不是“你可以随便扫一下就算了”，而是：

> **代码必须经过安全角度的正式审查。**

------

## 3）在 assessment 里怎么用

你不能只看：

- 有没有工具

还要看：

- 有没有正式审查结果
- 有没有结论
- 有没有处理 critical/high

------

## 4）主要涉及哪些 rules

- **API8-R2**

------

# 七、Inventory / Registry 是什么

这块是治理里最核心、最常被忽略的概念。

------

## 1）最简单定义

**API Inventory / Registry** 就是：

> 企业对自己有哪些 API 做的正式登记册。

------

## 2）通俗理解

你可以把它理解成：

> “公司的 API 花名册 + 档案系统。”

里面不只是名字，还要有很多关键信息。

------

## 3）为什么企业一定要有它

因为如果连“自己有哪些 API”都说不清，就根本谈不上：

- 安全治理
- owner 责任
- 版本管理
- 数据分类
- 暴露面控制
- 下线和整改

------

## 4）一个好的 registry 里通常会有什么

你前面 rule API9-R1 已经给了方向，常见包括：

- API IT owner
- API Business owner
- API description
- version
- status（active / deprecated / inactive）
- managed business service/data
- data classification
- exposition level（internet/extranet/intranet）
- scope(s)
- authorized client applications

------

## 5）在 assessment 里怎么用

你要问：

- 这个项目的 API 是否在 registry 里
- owner 是否清楚
- classification 是否定义了
- authorized clients 是否登记了
- scope 是否可追踪
- status 是否受治理

------

## 6）主要涉及哪些 rules

- **API9-R1**：最直接
- 也会支持：
  - API2-R10
  - API5-R1
  - API0-R3
  - API3-R3

------

# 八、Ownership（责任人）是什么

------

## 1）为什么要有 owner

因为安全问题最后一定会落到：

- 谁负责改
- 谁批准上线
- 谁负责下线
- 谁解释业务合理性
- 谁对 API 分类和暴露负责

如果没有 owner，很多 rule 就算写着也很难真正执行。

------

## 2）常见两类 owner

### IT Owner

更偏技术和实现责任。

### Business Owner

更偏业务目标、业务数据和使用场景责任。

------

## 3）为什么两者都要有

因为 API 安全很多问题不是纯技术问题。

### 例如

- 这个字段能不能对外暴露
- 这个 API 是否还需要保留
- 这个 scope 是否合理
- 这个 partner 是否应该继续被授权

这些都需要业务侧参与。

------

## 4）在 assessment 里怎么用

你要问：

- 谁是这个 API 的 IT owner
- 谁是 business owner
- 当发现 non-compliance 时谁负责处理
- 当 API 6 个月不用时谁决定停用

------

## 5）主要涉及哪些 rules

- **API9-R1**
- **API9-R2**

------

# 九、Unused API 是什么，为什么规则会要求下线

------

## 1）最简单定义

**Unused API** 就是长期没人用的 API。

------

## 2）为什么不用的 API 反而危险

很多人直觉会觉得：

- 没人用就没风险

其实相反，没人在意的 API 很容易变成：

- 漏洞没人修
- 版本没人管
- 权限还留着
- 文档过时
- owner 不明确
- 被攻击了也没人关注

所以“长期不用但仍暴露着”的 API，其实是治理上的风险点。

------

## 3）为什么规则会说 6 个月没用就该停用

因为企业治理要减少“僵尸暴露面”。

### 通俗理解

不需要的门，就应该关掉，不要一直开着。

------

## 4）在 assessment 里怎么用

你要问：

- 是否有 API usage 数据
- 是否有 inactive / deprecated 生命周期管理
- 长期无流量 API 是否会被 review
- 是否有 owner 来决定保留或下线

------

## 5）主要涉及哪些 rules

- **API9-R2**

------

# 十、Major Release 前测试，为什么会被写进规则

------

## 1）为什么 major release 前必须重新测

因为大版本改动通常意味着：

- 接口变化
- 配置变化
- 依赖变化
- 暴露面变化
- 身份授权逻辑变化

也就是说：

- 旧测试结果不能自动证明新版本也安全

------

## 2）在 assessment 里怎么理解

你要问：

- 这个项目每次 major release 前是否重新做漏洞测试 / pentest
- 只是“以前测过”不够
- 当前版本上线前有没有重新验证

------

## 3）主要涉及哪些 rules

- **API7-R6**
- **API7-R7**
- **API8-R2**

------

# 十一、Critical / High 漏洞为什么必须在 go-live 前修复

------

## 1）最简单理解

因为这类问题意味着：

- 风险已知
- 严重性已知
- 如果还带着上线，就是企业主动接受高风险

------

## 2）为什么规则要写死这件事

为了防止项目团队说：

- “先上线再说”
- “后面有空再修”
- “反正只是个扫描结果”

企业规则会用这种方式把红线写清楚。

------

## 3）在 assessment 里怎么用

你要问：

- 当前版本上线前是否还有 unresolved critical/high
- 是真实 risk accepted，还是还没处理完
- 有没有正式的例外批准流程

------

## 4）主要涉及哪些 rules

- **API7-R6**
- **API7-R7**
- **API8-R2**

------

# 十二、SAST / DAST / Pentest / Inventory / Ownership 放在一起怎么理解

这一步非常重要。

你可以把它们理解成一个闭环：

### 第一步：知道自己有什么

- Inventory
- Registry
- Ownership

### 第二步：知道应该怎么保护

- Rules
- Classification
- Scope / Exposure / Client mapping

### 第三步：知道是否真的做到了

- SAST
- DAST
- Pentest

### 第四步：知道发现问题后谁负责

- IT owner
- Business owner
- remediation workflow

### 第五步：知道没用的东西会不会收掉

- Unused API decommission

这就是企业安全治理的基本闭环。

------

# 十三、这一章在项目里怎么真正检查

这部分最实用。

------

## 1）查 SAST / Code Audit

看：

- Fortify / SAST 报告
- 上线前安全审查记录
- CI/CD 是否跑扫描
- critical/high 是否闭环

------

## 2）查 DAST / Vulnerability Testing

看：

- 动态扫描报告
- API 安全扫描报告
- 生产/预发的定期检测记录
- 月度/版本前测试记录

------

## 3）查 Pentest

看：

- 渗透测试报告
- 范围
- 时间
- 发现项和整改状态
- 是否为 gray-box
- 是否覆盖 internet-facing API

------

## 4）查 Inventory / Registry

看：

- API registry 平台
- Confluence/平台登记
- Apigee API product / catalog
- owner / classification / scope / status 是否完整

------

## 5）查 Usage / Lifecycle

看：

- 流量统计
- 长期无调用 API 清单
- deprecation / decommission 流程
- inactive API 是否关掉

------

# 十四、这一章和 rules 的对应关系

------

## SAST / Code Audit

- **API8-R2**

------

## Vulnerability Testing

- **API7-R6**

------

## Pentest

- **API7-R7**

------

## Governance / Registry / Ownership

- **API9-R1**

------

## Unused API Deactivation

- **API9-R2**

------

# 十五、一个完整例子，把这一章串起来

假设你在评估一个互联网暴露的客户 API。

### 第一步：你先查 registry

确认：

- 这个 API 有没有登记
- owner 是谁
- 分类是什么
- scope 和 authorized clients 是谁

### 第二步：你查安全测试

确认：

- 当前版本上线前有没有做 SAST
- 有没有 vulnerability testing
- 有没有 pentest
- critical/high 是否修复

### 第三步：你查生命周期

确认：

- 这个 API 还是 active 吗
- 最近 6 个月有没有使用
- 如果没用，为什么还留着

### 第四步：你下 assessment 结论

这时你不只是说“代码写得行不行”，而是在说：

> 这个 API 有没有被企业安全流程真正纳入治理。

------

# 十六、这一章最重要的 10 句话

## 第一句

**安全不是“我们觉得自己挺安全”，而是要有验证和治理证据。**

## 第二句

**SAST 看代码，DAST 看运行时，Pentest 看攻击路径。**

## 第三句

**SAST、DAST、Pentest 不是替代关系，而是互补关系。**

## 第四句

**Code audit 的重点不只是“跑过工具”，而是“问题是否被正式审查并处理”。**

## 第五句

**API inventory/registry 是治理的起点，没有清单就很难治理。**

## 第六句

**Owner 不清楚，很多安全要求最后就没人真正负责。**

## 第七句

**不用的 API 不是“没风险”，而是“容易变成僵尸暴露面”。**

## 第八句

**Major release 前重新测试，是因为旧结论不能自动继承到新版本。**

## 第九句

**Critical / High 漏洞若未修复就上线，是非常明确的治理风险。**

## 第十句

**做 assessment 时，既要看技术实现，也要看是否纳入了测试与治理闭环。**

------

## 十七、一句话总结这章

你以后看到一个 API，不要只问“它写得怎么样”，还要问：
 **它有没有被登记、被测试、被负责、被持续管理。**



# 第 17 部分：把所有概念串成“怎么做项目 assessment”

这一章你先抓住一句话：

> **做 assessment，不是从 rule 直接跳到结论，而是要走完“架构 → 调用链 → 控制点 → 证据 → 结论”这条路。**

------

## 一、先说最终目标是什么

你做一个 project assessment，最终不是为了写很多安全术语，而是为了回答这 4 个问题：

### 1. 这条 rule 到底要求什么

不是照抄原文，而是翻译成人话。

### 2. 这条 rule 应该由系统哪一层负责

是 WebSSO、前端、BFF、gateway、backend，还是治理流程。

### 3. 这个项目里有没有相应实现

也就是代码、配置、平台策略、测试、文档里有没有证据。

### 4. 基于现有证据，最后怎么判

- compliant
- partially compliant
- non-compliant
- unknown / evidence missing

------

## 二、Assessment 的总流程，你以后固定按这个走

你后面无论评估哪个项目，都建议按这个顺序。

------

### 第一步：先画出最小调用链

你要先搞清楚：

- User 是谁
- Client 是谁
- 有没有前端
- 有没有 BFF
- WebSSO 在哪里
- Gateway 是谁
- Backend 是谁
- 有没有下游服务

### 你要得到的最小图

哪怕只是脑子里，也要能说出来：

**User → Client →（BFF?）→ Gateway → Backend → Downstream**

以及登录链：

**Client ↔ WebSSO / OIDC Provider**

### 为什么这一步最重要

因为如果你连“谁在调用谁”都没理清，后面所有 rule 都会看错层。

------

### 第二步：判断每条 rule 属于哪一层

这是整个 assessment 最值钱的能力。

你一定要先学会这件事：

> **不是所有 rule 都去 Apigee 找，也不是所有 rule 都去代码里找。**

------

#### 典型分层

### A. 身份与登录层

看：

- WebSSO
- OIDC/OAuth 配置
- flow
- token issuance
- MFA
- SSO/session

常见相关 rule：

- API0-R2
- API2-R1
- API2-R4
- API2-R6
- API2-R7
- API2-R8
- API2-R9
- API6-R1
- API7-R5

------

### B. Gateway / Exposition 层

看：

- Apigee / Axway / API Gateway
- route
- product
- policy
- rate limit
- CORS
- headers
- error baseline
- external exposure

常见相关 rule：

- API0-R3
- API4-R1
- API7-R1
- API7-R2
- API7-R3
- API7-R4
- API8-R1
- API10-R1（部分）

------

### C. Backend / Business Service 层

看：

- Spring Boot 代码
- service logic
- security config
- DTO
- object ownership check
- field filtering
- business validation

常见相关 rule：

- API1-R1
- API3-R1
- API3-R2
- API3-R4
- API5-R1
- API8-R1

------

### D. Client / Frontend / BFF 层

看：

- Angular auth flow
- token storage
- session / cookie
- redirect handling
- BFF credential custody
- logout behavior

常见相关 rule：

- API0-R2
- API2-R1
- API2-R5
- API2-R8

------

### E. 治理与流程层

看：

- registry
- owner
- test reports
- SAST/DAST/pentest
- lifecycle
- unused API decommission

常见相关 rule：

- API7-R6
- API7-R7
- API8-R2
- API9-R1
- API9-R2

------

## 三、第三步：按“证据类型”去找，而不是盲看代码

你以后不要一上来就翻 Java 代码。
 应该先想：

> **这条 rule，最可能在哪类证据里体现？**

------

## 1）架构图证据

适合看：

- API0-R3：流量是不是先过 gateway
- API0-R2：有没有 BFF、WebSSO、Client 类型
- API3-R3：technical API 是否对外暴露
- 系统角色分层是否合理

------

## 2）配置证据

适合看：

- token endpoint / issuer / client_id
- TLS / SSL bundle / JKS
- CORS allowlist
- rate limit / quota
- token TTL
- session timeout
- API base URL / gateway domain

------

## 3）代码证据

适合看：

- grant flow 实现
- token 获取方式
- object-level authorization
- `@PreAuthorize`
- DTO / response trimming
- validation
- error handler
- logging

------

## 4）平台证据

适合看：

- Apigee policy
- API product
- app registration
- WebSSO client registration
- certificate setup
- K8S ingress / service exposure

------

## 5）文档与报告证据

适合看：

- OpenAPI / Swagger
- SAST report
- DAST report
- pentest report
- API registry
- ownership / classification / exposure level

------

## 四、第四步：对每条 rule，用固定问题模板去问

这是最实用的部分。
 你以后评估一条 rule，就按下面这 5 个问题走。

------

### 1. 这条 rule 在要求什么

先说成人话。

例如：

#### API0-R3

外部 API 必须通过 gateway 暴露。

#### API2-R8

会话必须安全管理，logout 要真失效，超时要合理。

#### API3-R1

只给必要数据，不要过度暴露。

------

### 2. 这条 rule 应该由哪一层负责

比如：

#### API0-R3

主要是 gateway / exposure layer。

#### API2-R8

主要是 WebSSO / BFF / frontend / backend session 配合。

#### API3-R1

主要是 backend，gateway 有时辅助做 response filtering。

------

### 3. 项目里看到了什么实现

这里不能空泛。

例如不要只写：

- “项目用了 Apigee”

而要写：

- “External ingress routes only through Apigee gateway”
- “Swagger server URL points to gateway domain”
- “No direct backend exposure identified”
- “Gateway policies include rate limiting and CORS control”

------

### 4. 还有什么缺口或证据不足

这一步很关键。

例如：

- 没看到 backend 是否还有旁路暴露
- 没看到 refresh token policy
- 没看到 logout 是否触发 revoke
- 没看到 object-level authorization 代码证据
- 没看到 registry 中的 owner/classification

------

### 5. 最后怎么判

然后才下结论。

------

## 五、最常用的 4 档结论，你要会用

我建议你后面默认用这 4 档，最稳。

### 1. Compliant

要求清楚，证据充分，关键控制点已落地。

### 2. Partially Compliant

方向对、做了一部分，但缺关键细节、范围不完整或证据不够闭环。

### 3. Non-Compliant

明显没做，或与 rule 要求冲突，或存在明确高风险缺口。

### 4. Unknown / Insufficient Evidence

当前没有足够证据，不应硬判。

------

## 六、什么情况下最容易判成 Partial

这类情况在企业 assessment 里非常常见。

### 1）有技术，但不是完整控制

例如：

- 用了 OAuth2，但没证明 browser/BFF 凭证边界符合要求
- 有 gateway，但 backend 仍可能直连暴露
- 有 role 检查，但没看到 object-level authorization

### 2）有文档，但没看到项目证据

例如：

- Apigee guideline 写得很好
- 但没看到当前 API product / policy 真的启用

### 3）有一层做了，另一层没做

例如：

- gateway 做了 coarse-grained auth
- backend 没做 fine-grained auth

### 4）实现可能有，但证据不够

例如：

- 说“应该有日志”
- 但没看到日志内容、集中化或 retention 证据

------

## 七、什么情况下不要急着判 Compliant

这个特别重要。
 企业里最容易高估合规性。

你看到这些情况时，要谨慎：

### 1）“项目用了某个技术”

例如：

- 用了 WebSSO
- 用了 Apigee
- 用了 JWT
- 用了 Spring Security

这些都只是起点，不是结论。

------

### 2）“代码里有相关关键词”

例如：

- 出现了 `client_id`
- 出现了 `Authorization`
- 出现了 `@PreAuthorize`
- 出现了 `CORS config`

你还得看：

- 用得对不对
- 范围够不够
- 有没有绕过
- 和 rule 是不是一一对应

------

### 3）“文档里写了应该这么做”

文档不是证据的全部。
 你还要看当前项目有没有真的这么配置。

------

## 八、把所有章节串成一张 assessment 思维图

现在把前面所有概念收成一个完整脑图。

------

## 1）先看系统角色

你先问：

- 谁是 User
- 谁是 Client
- 是 public 还是 confidential client
- WebSSO 在哪
- Gateway 在哪
- Backend 在哪
- 有没有 BFF

这一步对应前面：

- 第 1 章
- 第 2 章

------

## 2）再看登录和取 token

你再问：

- 用的是 OAuth2/OIDC 吗
- 是什么 flow
- client_id 是谁
- user identity 怎么体现
- 有没有 PKCE
- redirect_uri 怎么处理
- nonce / state 怎么处理

这一步对应：

- 第 3 到 8 章
- 第 11 章

------

## 3）再看 token 怎么管

你再问：

- access token / ID token / refresh token 各是什么
- 生命周期多长
- 格式是什么
- 存哪
- 怎么传
- 能不能 revoke / rotate

这一步对应：

- 第 6 到 10 章

------

## 4）再看 gateway 负责什么

你再问：

- 外部流量是不是都过 gateway
- token 基础校验有没有
- rate limit 有没有
- CORS 有没有
- error/header 基线有没有
- 是否暴露了不该暴露的 path

这一步对应：

- 第 12、13、14 章

------

## 5）再看 backend 负责什么

你再问：

- object-level authorization 有没有
- field-level authorization 有没有
- need-to-know 有没有
- secret data 会不会暴露
- semantic validation 有没有

这一步对应：

- 第 13、14 章

------

## 6）最后看运行与治理

你再问：

- 日志能不能追踪
- correlation id 有没有
- SIEM / retention 有没有
- SAST / DAST / pentest 有没有
- owner / registry / decommission 有没有

这一步对应：

- 第 15、16 章

------

## 九、给你一个真正可执行的项目 assessment 顺序

这部分你以后可以直接照着做。

------

### 第 1 步：拿到基本材料

先收集：

- 架构图
- API 文档 / Swagger
- 前后端配置
- gateway 信息（Apigee/Axway）
- WebSSO / OIDC 信息
- 关键代码位置
- 安全测试报告
- API registry / owner 信息

------

### 第 2 步：先画最小调用链

你自己先写一行出来：

- 用户怎么登录
- 前端调谁
- BFF 有没有
- gateway 在哪
- backend 在哪
- token 从哪来

------

### 第 3 步：按大类分 rule

你已经有了分类：

- Design
- Network
- Token
- Client
- Authorization
- Protection
- Others / Governance

先按大类评估，比一条条乱跳更不容易乱。

------

### 第 4 步：每条 rule 先定“主负责层”

例如：

#### API0-R3

主负责层：gateway / exposure

#### API2-R8

主负责层：WebSSO + app session / BFF

#### API3-R1

主负责层：backend

#### API9-R1

主负责层：governance / registry

这样你就知道去哪找证据。

------

### 第 5 步：写“Observed evidence”

你写证据时尽量具体，不要空话。

#### 不好

- “Project uses OAuth”
- “Project has gateway”
- “Security is implemented”

#### 更好

- “Client obtains access token through OAuth/OIDC flow against enterprise WebSSO”
- “Gateway domain is used as the external API entry point”
- “Backend validates object ownership before returning resource data”
- “OpenAPI definition exposes gateway URLs rather than backend endpoints”
- “No evidence of refresh token usage in browser-based client”

------

### 第 6 步：写“Gap / limitation”

你要习惯主动写缺口，不要只写优点。

例如：

- “No evidence found regarding refresh token rotation”
- “No evidence found that backend direct exposure is technically blocked”
- “No evidence found for object-level authorization beyond role checks”
- “Insufficient evidence regarding centralized log retention and anti-tamper controls”

------

### 第 7 步：最后才给 coverage / compliance

到这一步再下：

- Full / Covered
- Partial / Partially Covered
- None / Not Covered

或者：

- Compliant
- Partially Compliant
- Non-Compliant
- Unknown

------

## 十、一个完整的小例子：你怎么评一条 rule

我们拿 **API1-R1 Fine-grained authorizations** 举例。

------

### 1. Rule intent

不仅 gateway 要做 coarse-grained auth，backend 还必须做 fine-grained auth，基于 user/client identity 判定对象和数据访问权限。

### 2. Relevant layer

主看：

- backend business service
   辅看：
- gateway scope control

### 3. Evidence to look for

- token / gateway 是否传递 user/client identity
- backend 是否检查 object ownership
- response 是否有 field-level filtering
- 是否只是 role check，没有 object-level check

### 4. Possible findings

#### 正向

- gateway verifies scopes
- backend checks current user against requested customer/account object
- response DTO trims sensitive/internal fields

#### 缺口

- only role-based checks found
- no evidence of object ownership validation
- no evidence of field-level response restriction

### 5. Conclusion

如果只有 role/scope，没有 object-level/field-level 证据，通常只能判：

- **Partially Compliant**

你看，这就把前面几乎所有章节都用起来了。

------

## 十一、你以后最容易犯的 6 个 assessment 错误

我提前帮你拦一下。

------

### 错误 1：看到技术名词就当合规

例如：

- 有 JWT
- 有 Apigee
- 有 Spring Security
   不等于合规。

------

### 错误 2：只看一层

例如：

- 只看 gateway，不看 backend
- 只看前端，不看 WebSSO
- 只看代码，不看平台配置

------

### 错误 3：只看“能不能跑”

能跑不等于安全。
 合规看的是：

- 边界
- 生命周期
- 控制点
- 证据

------

### 错误 4：证据不足时硬判

企业里最稳的做法是：

- 证据不足就写 unknown / partial
- 不要靠猜补空白

------

### 错误 5：把文档当成实现

guideline、架构说明、设计图都很重要，但它们不是当前项目一定真的这么做了的证明。

------

### 错误 6：只会背规则原文，不会翻译成人话

你最后交付给人看的 assessment，一定要能回答：

- 这条 rule 在项目里到底意味着什么
- 我看了哪里
- 看到了什么
- 为什么这么判

------

## 十二、最后给你一个“通用 assessment 模板”

你以后可以直接套。

------

### Rule:

写 rule id + name

### Intent:

用人话写这条 rule 真正在要求什么

### Relevant control layer:

写：

- WebSSO / Client / Gateway / Backend / Governance

### Evidence reviewed:

写你看了什么：

- code
- config
- gateway policy
- OpenAPI
- logs
- reports
- registry

### Observed implementation:

写当前项目做了什么

### Gap / limitation:

写还缺什么、没看到什么、哪里证据不足

### Assessment:

写：

- Compliant / Partial / Non-Compliant / Unknown

------

## 十三、这一章最重要的 10 句话

## 第一句

**Assessment 不是从 rule 直接跳结论，而是要先分层、找证据、再判断。**

## 第二句

**先画调用链，再评规则，会比直接逐条硬看稳定得多。**

## 第三句

**不是所有 rule 都去同一层找答案。**

## 第四句

**有技术存在，不等于 rule 已落地。**

## 第五句

**文档、代码、配置、平台、报告，都是证据，但分量不同。**

## 第六句

**Gateway、WebSSO、Backend、Governance 各自负责不同类型的 rule。**

## 第七句

**Partial 是很正常的结论，不代表你分析得不好，反而说明你在认真分层。**

## 第八句

**Unknown/Insufficient Evidence 比靠猜更专业。**

## 第九句

**好的 assessment 一定能说清楚：这条 rule 要求什么、项目在哪里实现、证据是什么、缺口是什么。**

## 第十句

**你真正的目标不是背术语，而是把“规则—控制层—证据—结论”这条链跑通。**

------

## 十四、整套内容的一句话总总结

你现在可以把整个 API security assessment 理解成：

> **先搞清系统里谁在跟谁说话、怎么登录、怎么拿 token、怎么进 gateway、怎么进 backend、怎么限制对象和字段、怎么记录和监控、怎么测试和治理，然后把这些实现一一映射到具体 rules 上，最后基于证据下结论。**
